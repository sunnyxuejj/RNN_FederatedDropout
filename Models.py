#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8


import torch.nn.functional as F
import torch
from torch import nn

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop


class RnnLm(nn.Module):
    def __init__(self, config, nvocab):
        super(RnnLm, self).__init__()
        self.args = config
        if not config.tied:
            self.embed = nn.Embedding(nvocab, config.d_embed)
        if config.rnn_type == 'GRU':
            self.encoder = nn.GRU(config.d_embed, config.rnn_hidden, config.rnn_layers,
                                  dropout=config.dropout, bias=True, bidirectional=False)
        elif config.rnn_type == 'LSTM':
            self.encoder = nn.GRU(config.d_embed, config.rnn_hidden, config.rnn_layers,
                                  dropout=config.dropout, bias=True, bidirectional=False)
        self.fc1 = nn.Linear(config.rnn_hidden, nvocab, bias=True)

    def get_embedded(self, word_indexes):
        if self.args.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embed(word_indexes)

    def forward(self, sents):
        embedded_sents = self.get_embedded(sents)
        out_sequence, _ = self.encoder(embedded_sents)
        out = self.fc1(out_sequence)
        out = out.view((out.size(0) * out.size(1), out.size(2)))
        return F.log_softmax(out, dim=1)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, config, ntoken):
        # def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.args = config
        self.rnn_type = config.rnn_type
        self.d_embed = config.d_embed
        self.rnn_hidden = config.rnn_hidden
        self.rnn_layers = config.rnn_layers
        self.dropout = config.dropout
        self.dropout_emd = config.dropout_emd  # 删掉了dropouti dropout for input embedding layers
        self.dropout_hh = config.dropout_hh
        self.tie_weights = config.tie_weights
        self.encoder = nn.Embedding(ntoken, self.d_embed)
        self.drop = nn.Dropout(p=self.dropout)
        self.lockdrop = LockedDropout()
        if self.args.gpu != -1:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        assert self.rnn_type in ['LSTM', 'GRU'], 'RNN type is not supported'

        if self.rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(self.d_embed if l == 0 else self.rnn_hidden, self.rnn_hidden if l != self.rnn_layers - 1 else (self.d_embed if self.tie_weights else self.rnn_hidden), 1, dropout=self.dropout) for l in range(self.rnn_layers)]
        if self.rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(self.d_embed if l == 0 else self.rnn_hidden, self.rnn_hidden if l != self.rnn_layers - 1 else self.d_embed, 1, dropout=self.dropout) for l in range(self.rnn_layers)]
        '''elif self.rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid,
                                   hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in
                         range(nlayers)]'''

        if self.dropout_hh and self.args.variational:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.dropout_hh) for rnn in self.rnns]

        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(self.rnn_hidden, ntoken)
        self.init_weights()

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        #  use weight tying on the embedding and softmax weight
        if self.tie_weights:
            self.decoder.weight = self.encoder.weight

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        raw_outputs = []
        outputs = []
        result = None
        if self.args.variational:
            emb = embedded_dropout(self.encoder, input, dropout=self.dropout_emd if self.training else 0)
            emb = self.lockdrop(emb, self.dropout_emd)  # lockdrop并不涉及权重的减少问题

            raw_output = emb
            new_hidden = []
            for l, rnn in enumerate(self.rnns):
                if hidden[0][0].size(1) != raw_output.size(1):
                    hidden = self.init_hidden(raw_output.size(1))
                output = rnn(raw_output, hidden[l])
                raw_output = output[0]
                new_hidden.append(output[1])
                raw_outputs.append(raw_output)
                if l!= self.args.rnn_layers - 1:
                    raw_output = self.lockdrop(raw_output, self.dropout)
                    outputs.append(raw_output)
            hidden = new_hidden

            output = self.lockdrop(raw_output, self.dropout)  # 这里有问题需要改进下
            outputs.append(output)
            output = self.decoder(output)
            output = output.view((output.size(0) * output.size(1), output.size(2)))
            result = F.log_softmax(output, dim=1)

        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.rnn_hidden if l != self.rnn_layers - 1 else (self.d_embed if self.tie_weights else self.self.rnn_hidden)).zero_().to(self.device),
                    weight.new(1, bsz, self.rnn_hidden if l != self.rnn_layers - 1 else (self.d_embed if self.tie_weights else self.self.rnn_hidden)).zero_().to(self.device))
                    for l in range(self.rnn_layers)]
        elif self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.self.rnn_hidden if l != self.rnn_layers - 1 else (self.d_embed if self.tie_weights else self.self.rnn_hidden)).zero_().to(self.device)
                    for l in range(self.rnn_layers)]

