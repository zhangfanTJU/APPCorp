import torch.nn as nn
from module.layers import LSTM, drop_sequence_sharedmask


class WordEncoder(nn.Module):
    def __init__(self, input_size, config):
        super(WordEncoder, self).__init__()
        self.dropout = config.dropout_mlp

        self.word_lstm = LSTM(
            input_size=input_size,
            hidden_size=config.word_hidden_size,
            num_layers=config.word_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_input,
            dropout_out=config.dropout_hidden,
        )

    def forward(self, batch_inputs, batch_masks):
        # batch_inputs: sen_num x sent_len x embed_dim
        # batch_masks   sen_num x sent_len
        hiddens, _ = self.word_lstm(batch_inputs, batch_masks)  # sent_len x sen_num x hidden*2
        hiddens.transpose_(1, 0)  # sen_num x sent_len x hidden*2

        if self.training:
            hiddens = drop_sequence_sharedmask(hiddens, self.dropout)

        return hiddens
