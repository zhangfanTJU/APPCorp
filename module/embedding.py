import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from module.layers import drop_input_independent
from transformers import BertTokenizer, BertModel


class Embedding(nn.Module):
    def __init__(self, config, vocab):
        super(Embedding, self).__init__()
        self.dropout_embed = config.dropout_embed

        word_embed = np.zeros((vocab.word_size, config.word_dims), dtype=np.float32)
        self.word_embed = nn.Embedding(vocab.word_size, config.word_dims, padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_embed))

        extword_embed = vocab.load_pretrained_embs(config.glove_path)
        extword_size, word_dims = extword_embed.shape
        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

    def forward(self, word_ids, extword_ids):
        # word_ids: sen_num x sent_len
        # extword_ids: sen_num x sent_len

        word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = drop_input_independent(batch_embed, self.dropout_embed)  # sen_num x sent_len x embed_dim

        return batch_embed


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(config.dropout_mlp)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.bert = BertModel.from_pretrained(config.bert_path)

    def tokenize(self, words):
        tokens = ["[CLS]"]
        lens = [1]  # cls
        for word in words:
            tokens_ = self.tokenizer.tokenize(word)
            lens.append(len(tokens_))
            tokens.extend(tokens_)
        lens.append(1)
        tokens.append("[SEP]")
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids, lens

    def merge_embedding(self, hiddens, lens, max_sent_len, merge='mean'):
        # hiddens: bert_len x 768
        # lens: list of word_len
        # lens include cls and sep

        bert_len, sent_len = sum(lens), len(lens)
        hiddens = hiddens[:bert_len]
        if bert_len != sent_len:  # no subword
            hiddens_split = list(torch.split(hiddens, lens, dim=0))
            if merge == 'mean':
                hiddens_split = list(map(lambda x: torch.mean(x, dim=0), hiddens_split))
            elif merge == 'sum':
                hiddens_split = list(map(lambda x: torch.sum(x, dim=0), hiddens_split))

            hiddens = torch.stack(hiddens_split)  # sent_len x 768

        hiddens = hiddens[1:-1]  # remove cls, sep
        sent_len -= 2

        if sent_len < max_sent_len:
            hiddens = F.pad(hiddens, (0, 0, 0, max_sent_len - sent_len))  # max_sent_len x 768

        return hiddens

    def forward(self, input_ids, attention_mask, batch_sent_lens):
        # input_ids: sen_num x bert_len
        # sent_lens: sen_num x list of word_len

        batch_hiddens, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # sen_num x bert_len x 768

        sent_lens = [sum(lens) - 2 for lens in batch_sent_lens]
        max_sent_len = max(sent_lens)

        sent_hiddens = list(torch.split(batch_hiddens, 1, dim=0))
        for i in range(len(sent_hiddens)):
            sent_hiddens[i] = self.merge_embedding(sent_hiddens[i].squeeze(0), batch_sent_lens[i], max_sent_len,
                                                   merge='mean')  # max_sent_len x 768

        batch_hiddens = torch.stack(sent_hiddens)  # sen_num x max_sent_len x 768

        if self.training:
            batch_hiddens = self.dropout(batch_hiddens)

        return batch_hiddens
