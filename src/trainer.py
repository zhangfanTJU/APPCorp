import gc
import time
import json
import torch
import logging
import subprocess
import numpy as np
import torch.nn as nn

from sklearn.metrics import classification_report

from src.loader import get_examples, data_iter
from src.optimizer import Optimizer
from src.utils import get_score, reformat


class Trainer():
    def __init__(self, model, config, vocab):
        self.model = model
        self.config = config
        self.report = False

        self.train_data = get_examples(self.config.train_file, model.embedding, vocab)
        self.batch_num = int(np.ceil(len(self.train_data) / float(self.config.train_batch_size)))

        self.dev_data = get_examples(self.config.dev_file, model.embedding, vocab)
        self.test_data = get_examples(self.config.test_file, model.embedding, vocab)

        # criterion
        weight = torch.FloatTensor(vocab.label_weights)
        weight = len(self.train_data) / weight
        weight = weight / torch.sum(weight)

        if config.use_cuda:
            weight = weight.to(config.device)

        self.criterion = nn.CrossEntropyLoss(weight)

        # label name and
        self.target_names = vocab._id2label

        # id2word of id2token
        if config.emb == 'bert':
            self.id2word = model.embedding.tokenizer._convert_id_to_token
        else:
            self.id2word = lambda id: vocab._id2word[id]

        # optimizer
        self.optimizer = Optimizer(model, config, self.batch_num)

        # count
        self.step = 0
        self.early_stop = -1
        self.best_train_f1, self.best_dev_f1, self.best_test_f1 = 0, 0, 0
        self.last_epoch = config.epochs + 1

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            gc.collect()
            train_f1 = self._train(epoch)
            self.logging_gpu_memory()

            gc.collect()
            dev_f1 = self._eval(epoch, "dev")
            self.logging_gpu_memory()

            gc.collect()
            test_f1 = self._eval(epoch, "test")
            self.logging_gpu_memory()

            # if self.best_dev_f1 < dev_f1 or (self.best_dev_f1 == dev_f1 and self.best_test_f1 < test_f1):
            if self.best_dev_f1 <= dev_f1 and self.best_test_f1 < test_f1:
                logging.info("Exceed history dev = %.2f, current train = %.2f dev = %.2f test = %.2f epoch = %d" %
                             (self.best_dev_f1, train_f1, dev_f1, test_f1, epoch))
                if self.config.save and epoch > self.config.save_after:
                    torch.save(self.model.state_dict(), self.config.save_model + str(epoch) + '.bin')

                self.best_train_f1 = train_f1
                self.best_dev_f1 = dev_f1
                self.best_test_f1 = test_f1
                self.early_stop = 0
            else:
                self.early_stop += 1
                if self.early_stop == self.config.early_stops:
                    logging.info("Eearly stop in epoch %d, best train: %.2f, dev: %.2f, test: %.2f" %
                                 (epoch - self.config.early_stops, self.best_train_f1, self.best_dev_f1, self.best_test_f1))
                    self.last_epoch = epoch
                    break

    def test(self):
        self.model.load_state_dict(torch.load(self.config.save_model, map_location=self.config.device))
        self._eval(-1, "test", test_batch_size=self.config.test_batch_size)

    def _train(self, epoch):
        self.optimizer.zero_grad()
        self.model.train()

        start_time = time.time()
        epoch_start_time = time.time()
        overall_losses = 0
        losses = 0
        batch_idx = 1
        y_pred = []
        y_true = []
        for batch_data in data_iter(self.train_data, self.config.train_batch_size, self.config, shuffle=True):
            torch.cuda.empty_cache()
            batch_inputs, batch_labels = self.batch2tensor(batch_data)

            batch_outputs, attens = self.model(batch_inputs)
            loss = self.criterion(batch_outputs, batch_labels)
            loss = loss / self.config.update_every
            loss.backward()

            loss_value = loss.detach().cpu().item()
            losses += loss_value
            overall_losses += loss_value

            y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
            y_true.extend(batch_labels.cpu().numpy().tolist())

            if batch_idx % self.config.update_every == 0 or batch_idx == self.batch_num:
                nn.utils.clip_grad_norm_(self.optimizer.all_params, max_norm=self.config.clip)
                for optimizer, scheduler in zip(self.optimizer.optims, self.optimizer.schedulers):
                    optimizer.step()
                    scheduler.step()
                self.optimizer.zero_grad()

                self.step += 1

            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                lrs = self.optimizer.get_lr()
                logging.info('| epoch {:3d} | step {:3d} | batch {:3d}/{:3d} | lr{} | loss {:.4f} | s/batch {:.2f}'.format(
                    epoch, self.step, batch_idx, self.batch_num, lrs, losses / self.config.log_interval,
                    elapsed / self.config.log_interval))

                start_time = time.time()
                losses = 0

            batch_idx += 1

        overall_losses /= self.batch_num
        during_time = time.time() - epoch_start_time

        # reformat
        overall_losses = reformat(overall_losses, 4)
        score, f1 = get_score(y_true, y_pred)

        logging.info('| epoch {:3d} | score {} | f1 {} | loss {} | time {:.2f}'.format(epoch, score, f1, overall_losses, during_time))
        if set(y_true) == set(y_pred) and self.report:
            report = classification_report(y_true, y_pred, digits=4, target_names=self.target_names)
            logging.info('\n' + report)

        return f1

    def _eval(self, epoch, data_nane, test_batch_size=None):
        self.model.eval()

        start_time = time.time()

        if data_nane == "dev":
            data = self.dev_data
        elif data_nane == "test":
            data = self.test_data
        else:
            Exception("No name data.")

        if test_batch_size is None:
            test_batch_size = self.config.test_batch_size

        y_pred = []
        y_true = []
        all_doc_words = []
        all_word_attens = []
        all_sent_attens = []
        with torch.no_grad():
            for batch_data in data_iter(data, test_batch_size, self.config, shuffle=False):
                torch.cuda.empty_cache()
                batch_inputs, batch_labels = self.batch2tensor(batch_data)

                batch_outputs, attens = self.model(batch_inputs)

                y_pred.extend(torch.max(batch_outputs, dim=1)[1].cpu().numpy().tolist())
                y_true.extend(batch_labels.cpu().numpy().tolist())

                if data_nane == "test":
                    doc_words = self.batch2word(batch_data)
                    word_attens, sent_attens = attens
                    # self.check_lens(doc_words, word_attens, sent_attens)
                    all_doc_words.extend(doc_words)
                    if word_attens is not None:
                        all_word_attens.extend(word_attens)
                    all_sent_attens.extend(sent_attens)

            score, f1 = get_score(y_true, y_pred)

            during_time = time.time() - start_time
            logging.info('| epoch {:3d} | {} | score {} | f1 {} | time {:.2f}'.format(epoch, data_nane, score, f1, during_time))
            if set(y_true) == set(y_pred) and self.report:
                report = classification_report(y_true, y_pred, digits=4)
                logging.info('\n' + report)

        if self.config.save and data_nane == "test":
            words = self.convert_sort(all_doc_words)
            if len(all_word_attens) > 0:
                all_word_attens = self.convert_sort(all_word_attens)
            all_sent_attens = self.convert_sort(all_sent_attens)
            y_true = self.convert_sort(y_true)
            y_pred = self.convert_sort(y_pred)

            file = open(self.config.save_test + str(epoch) + '.json', 'w')
            dic = {
                'words': words,
                'word_attens': all_word_attens,
                'sent_attens': all_sent_attens,
                'ture_label': y_true,
                'pred_label': y_pred
            }
            file.write(json.dumps(dic))
            file.close()

        return f1

    def convert_sort(self, lst):
        return [lst[self.config.sorted_indices.index(idx)] for idx in range(len(self.config.sorted_indices))]

    def batch2tensor(self, batch_data):
        if self.config.emb == 'bert':
            return self.batch2tensor_bert(batch_data)
        else:
            return self.batch2tensor_glove(batch_data)

    def batch2tensor_glove(self, batch_data):
        '''
            [[label, doc_len, [[graph, [word_id0, ...], [extword_id0, ...]], ...]]
        '''

        batch_size = len(batch_data)
        batch_labels = []
        batch_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            batch_labels.append(doc_data[0])
            batch_lens.append(doc_data[1])
            sent_lens = [len(sent_data[2]) for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        sent_num = sum(batch_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs1 = torch.zeros((sent_num, max_sent_len), dtype=torch.int64)
        batch_inputs2 = torch.zeros((sent_num, max_sent_len), dtype=torch.int64)

        batch_masks = torch.zeros((sent_num, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(batch_labels)

        batch_graphs = []
        sent_idx = 0
        for b in range(batch_size):
            for idx in range(batch_lens[b]):
                sent_data = batch_data[b][2][idx]

                sent_len = len(sent_data[1])
                for word_idx in range(sent_len):
                    batch_inputs1[sent_idx, word_idx] = sent_data[1][word_idx]
                    batch_inputs2[sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[sent_idx, word_idx] = 1

                if self.config.use_graph:
                    batch_graphs.append(sent_data[0])

                sent_idx += 1

        if self.config.use_cuda:
            batch_inputs1 = batch_inputs1.to(self.config.device)
            batch_inputs2 = batch_inputs2.to(self.config.device)
            batch_masks = batch_masks.to(self.config.device)
            batch_labels = batch_labels.to(self.config.device)

        return (batch_inputs1, batch_inputs2, batch_graphs, batch_masks, batch_lens), batch_labels

    def batch2tensor_bert(self, batch_data):
        '''
            [[label, doc_len, [[graph, lens, [token_id0, ...]], ...]]
        '''

        batch_size = len(batch_data)
        batch_labels = []
        batch_lens = []
        batch_sent_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            batch_labels.append(doc_data[0])
            batch_lens.append(doc_data[1])
            batch_sent_lens.extend([sent_data[1] for sent_data in doc_data[2]])
            sent_lens = [len(sent_data[2]) for sent_data in doc_data[2]]
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        sent_num = sum(batch_lens)
        max_sent_len = max(doc_max_sent_len)

        batch_inputs = torch.zeros((sent_num, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((sent_num, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(batch_labels)

        batch_graphs = []
        sent_idx = 0
        for b in range(batch_size):
            for idx in range(batch_lens[b]):
                sent_data = batch_data[b][2][idx]

                sent_len = len(sent_data[2])
                for word_idx in range(sent_len):
                    batch_inputs[sent_idx, word_idx] = sent_data[2][word_idx]
                    batch_masks[sent_idx, word_idx] = 1

                if self.config.use_graph:
                    batch_graphs.append(sent_data[0])

                sent_idx += 1

        if self.config.use_cuda:
            batch_inputs = batch_inputs.to(self.config.device)
            batch_masks = batch_masks.to(self.config.device)
            batch_labels = batch_labels.to(self.config.device)

        return (batch_inputs, batch_sent_lens, batch_graphs, batch_masks, batch_lens), batch_labels

    def batch2word(self, batch_data):
        if self.config.emb == 'bert':
            return self.batch2word_bert(batch_data)
        else:
            return self.batch2word_glove(batch_data)

    def batch2word_bert(self, batch_data):
        """
            [[label, doc_len, [[graph, sent_lens, [token_id, ...]], ...]]
        """

        doc_words = []
        for doc_data in batch_data:
            sents_words = []
            sents_data = doc_data[2]
            for sent_data in sents_data:
                sent_words = [self.id2word(id) for id in sent_data[2]]
                sents_words.append(sent_words)
            doc_words.append(sents_words)
        return doc_words

    def batch2word_glove(self, batch_data):
        """
            [[label, doc_len, [[graph, [word_id, ...], [extword_id, ...]], ...]]
        """

        doc_words = []
        for doc_data in batch_data:
            sents_words = []
            sents_data = doc_data[2]
            for sent_data in sents_data:
                sent_words = [self.id2word(id) for id in sent_data[1]]
                sents_words.append(sent_words)
            doc_words.append(sents_words)
        return doc_words

    def check_lens(self, doc_words, word_attens, sent_attens):
        assert len(doc_words) == len(sent_attens)  # batch_size
        if word_attens is not None:
            assert len(doc_words) == len(word_attens)
            for doc_word, word_atten, sent_atten in zip(doc_words, word_attens, sent_attens):
                assert len(doc_word) == len(word_atten) == len(sent_atten)  # doc_len
                for sent_word, word_atten_ in zip(doc_word, word_atten):
                    assert len(sent_word) == len(word_atten_)  # sent_len
        else:
            for doc_word, sent_atten in zip(doc_words, sent_attens):
                assert len(doc_word) == len(sent_atten)  # doc_len

    def logging_gpu_memory(self):
        """
        Get the current GPU memory usage.
        Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
        Returns
        -------
        ``Dict[int, int]``
            Keys are device ids as integers.
            Values are memory usage as integers in MB.
            Returns an empty ``dict`` if GPUs are not available.
        """
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                encoding="utf-8",
            )
            info = [x.split(',') for x in result.strip().split("\n")]
            dic = {gpu: [int(mem[0]), int(mem[1])] for gpu, mem in enumerate(info)}
            gpu_id = self.config.gpu_id
            lst = dic[gpu_id]
            logging.info('| gpu id: {} | use {:5d}M / {:5d}M'.format(self.config.gpu_id, lst[0], lst[1]))

        except FileNotFoundError:
            # `nvidia-smi` doesn't exist, assume that means no GPU.
            return {}
        except:  # noqa
            # Catch *all* exceptions, because this memory check is a nice-to-have
            # and we'd never want a training run to fail because of it.
            logging.info("unable to check gpu_memory_mb(), continuing")
            return {}
