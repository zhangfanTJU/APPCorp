import torch
import logging
import numpy as np
import torch.nn as nn
from src.utils import reformat
import torch.nn.functional as F
from module.layers import NoLinear
from module import WordEncoder, SentEncoder, Attention, Embedding, BertEmbedding, GraphEncoder


class Model(nn.Module):
    def __init__(self, config, vocab):
        super(Model, self).__init__()
        self.vocab = vocab
        self.config = config
        self.emb = config.emb
        self.use_graph = config.use_graph

        self.sent_rep_size = config.word_hidden_size * 2
        self.doc_rep_size = config.sent_hidden_size * 2

        if self.emb == 'bert':
            self.embedding = BertEmbedding(config)
            self.proj = NoLinear(768, config.word_dims)
        else:
            self.embedding = Embedding(config, vocab)

        if self.use_graph:
            self.graph_encoder = GraphEncoder(config.word_dims, config, vocab)

        input_size = config.word_dims * 2 if self.use_graph else config.word_dims
        self.word_encoder = WordEncoder(input_size, config)
        self.word_attention = Attention(self.sent_rep_size)

        self.sent_encoder = SentEncoder(self.sent_rep_size, config)
        self.sent_attention = Attention(self.doc_rep_size)

        self.out = NoLinear(self.doc_rep_size, vocab.label_size, bias=True)

        if config.use_cuda:
            self.to(config.device)

        logging.info('Build model with {} embedding, graph {}.'.format(config.emb, config.use_graph))

        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
        para_need_grad = list(filter(lambda p: p.requires_grad, self.parameters()))
        para_need_grad_num = sum([np.prod(list(p.size())) for p in para_need_grad])
        logging.info('Model param num: %.2f M, need grad: %.2f M.' % (para_num / 1e6, para_need_grad_num / 1e6))

        # logging.info(self)

    def forward(self, batch_inputs):
        # batch_inputs: sen_num x sent_len
        # batch_masks : sen_num x sent_len

        if self.emb == 'bert':
            batch_inputs, batch_sent_lens, batch_graphs, batch_masks, batch_doc_lens = batch_inputs
            max_sent_len = batch_inputs.shape[1] - 2  # sen_num x sent_len

            batch_embed = self.embedding(batch_inputs, batch_masks, batch_sent_lens)  # sen_num x sent_len x sent_rep_size
            batch_embed = self.proj(batch_embed)

            batch_masks = batch_masks[:, 2:]
        else:
            batch_inputs1, batch_inputs2, batch_graphs, batch_masks, batch_doc_lens = batch_inputs
            max_sent_len = batch_inputs1.shape[1]  # sen_num x sent_len

            batch_embed = self.embedding(batch_inputs1, batch_inputs2)  # sen_num x word_dims

        batch_size, max_doc_len = len(batch_doc_lens), max(batch_doc_lens)

        if self.use_graph:
            batch_embed_gnn = self.graph_encoder(batch_embed, batch_graphs)  # sen_num x sent_len x sent_rep_size
            batch_inputs = torch.cat([batch_embed, batch_embed_gnn], dim=-1)
        else:
            batch_inputs = batch_embed

        batch_hiddens = self.word_encoder(batch_inputs, batch_masks)  # sen_num x sent_len x sent_rep_size

        sent_reps, word_scores = self.word_attention(batch_hiddens, batch_masks)  # sen_num x sent_rep_size

        if max_doc_len == min(batch_doc_lens):
            sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)  # b x doc_len x sent_rep_size
            batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
            if not self.training:
                word_scores = word_scores.view(batch_size, max_doc_len, max_sent_len)  # b x doc_len x max_sent_len
        else:
            batch_masks = self.pad_batch(batch_masks, batch_doc_lens)
            sent_reps = self.pad_batch(sent_reps, batch_doc_lens)
            if not self.training:
                word_scores = self.pad_batch(word_scores, batch_doc_lens)

        sent_masks = batch_masks.bool().any(2).float()  # b x doc_len

        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)  # b x doc_len x doc_rep_size
        doc_reps, sent_scores = self.sent_attention(sent_hiddens, sent_masks)  # b x doc_rep_size

        batch_outputs = self.out(doc_reps)  # b x num_labels

        if not self.training:
            if word_scores is not None:
                attens = self.merge_attens(word_scores, sent_scores, batch_masks)
            else:
                attens = self.merge_sent_attens(sent_scores, batch_masks)
        else:
            attens = None

        return batch_outputs, attens

    def pad_batch(self, batch_input, batch_doc_lens):
        batch_size, max_doc_len = len(batch_doc_lens), max(batch_doc_lens)
        batch_split = list(torch.split(batch_input, batch_doc_lens))

        for i in range(batch_size):
            doc_len = batch_doc_lens[i]
            if doc_len < max_doc_len:
                batch_split[i] = F.pad(batch_split[i], (0, 0, 0, max_doc_len - doc_len))

        batch_output = torch.stack(batch_split)
        return batch_output

    def merge_attens(self, word_scores, sent_scores, batch_masks):
        # word_scores: b x max_doc_len x max_sent_len
        # batch_masks: b x max_doc_len x max_sent_len
        # sent_scores: b x max_doc_len

        sent_attens = []
        word_attens = []
        # [[sent0_len, sent2_len, ...], ...]
        batch_lens = torch.sum(batch_masks, 2).int().cpu().tolist()  # b x max_doc_len

        word_scores = word_scores.cpu().tolist()
        sent_scores = sent_scores.cpu().tolist()
        for i, sent_lens in enumerate(batch_lens):
            word_attens_ = []
            sent_lens = list(filter(lambda x: x > 0, sent_lens))
            doc_len = len(sent_lens)
            for j, sent_len in enumerate(sent_lens):
                scores = word_scores[i][j][:sent_len]
                norm_scores = [reformat(score / sum(scores), 2) for score in scores]
                word_attens_.append(norm_scores)
            scores = sent_scores[i][:doc_len]
            norm_scores = [reformat(score / sum(scores), 2) for score in scores]
            sent_attens.append(norm_scores)
            word_attens.append(word_attens_)

        return word_attens, sent_attens

    def merge_sent_attens(self, sent_scores, batch_masks):
        # word_scores: b x max_doc_len x max_sent_len
        # batch_masks: b x max_doc_len x max_sent_len
        # sent_scores: b x max_doc_len

        sent_attens = []
        word_attens = None
        # [[sent0_len, sent2_len, ...], ...]
        batch_lens = torch.sum(batch_masks, 2).int().cpu().tolist()  # b x max_doc_len

        sent_scores = sent_scores.detach().cpu().tolist()
        for i, sent_lens in enumerate(batch_lens):
            sent_lens = list(filter(lambda x: x > 0, sent_lens))
            doc_len = len(sent_lens)

            scores = sent_scores[i][:doc_len]
            norm_scores = [reformat(score / sum(scores), 2) for score in scores]
            sent_attens.append(norm_scores)

        return word_attens, sent_attens
