#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import torch
import time

from torch import nn
from torch.utils.data import DataLoader, Dataset
from util import repackage_hidden
from torch.autograd import Variable
from dgc.compression import DGCCompressor


def quantizer(params):
    params_q = {}
    for k in params.keys():
        params_q[k] = torch.quantize_per_tensor(params[k], 0.01, 0, torch.qint8)
    return params_q


def sampling(params, dropout= 0.3):
    params_q = {}
    for k in params.keys():
        x = params[k]
        shape = x.shape
        m = x.data.new(shape).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        params_q[k] = mask * x
    return params_q


class DatasetSplitLM(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdateLM(object):
    def __init__(self, args, train_dataset, val_dataset, nround, user, data_num, memory=None):
        self.args = args
        self.round = nround
        self.user = user
        self.data_num = data_num
        self.loss_func = nn.CrossEntropyLoss()
        self.memory = memory
        self.compression = DGCCompressor(compress_ratio=self.warmup_compress_ratio(), memory=self.memory)
        self.data_loader = DataLoader(train_dataset, batch_size=self.args.local_bs, shuffle=True)
        self.valdata_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    def update_weights(self, net, lr, current_epoch):
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=self.args.momentum, weight_decay=self.args.wdecay)

        list_loss, list_pp, accs = [], [], []

        for iter in range(self.args.local_ep):
            total_loss = 0
            loss_int_sum = 0
            total_perp = 0
            corrent = 0
            word_num = 0
            start_time = time.time()
            begin_time = time.time()
            hidden = net.init_hidden(self.args.local_bs)
            for batch_ind, sents in enumerate(self.data_loader):
                lr2 = optimizer.param_groups[0]['lr']
                net.train()
                optimizer.zero_grad()
                #sents.sort(key=lambda l: len(l), reverse=True)
                x = torch.stack(sents[:-1])
                y = torch.stack(sents[1:]).view(-1)
                word_num += y.size(0)
                if self.args.gpu != -1:
                    net = net.cuda()
                    x, y = x.cuda(), y.cuda()
                hidden = repackage_hidden(hidden)
                out, hidden = net(x, hidden, return_h=False)
                raw_loss = self.loss_func(out, y)
                loss = raw_loss
                loss.backward()
                if self.args.clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
                optimizer.step()
                total_loss += raw_loss.data
                loss_int_sum += raw_loss.data
                prob = out.exp()[torch.arange(0, y.shape[0], dtype=torch.int64), y]
                perplexity = 2 ** prob.log2().neg().mean().item()
                total_perp = total_perp + perplexity
                list_pp.append(perplexity)
                optimizer.param_groups[0]['lr'] = lr2
                top_3, top3_index = torch.topk(out, 3, dim=1)
                for i in range(top3_index.size(0)):
                    if y[i] in top3_index[i]:
                        corrent += 1
                # Calculate perplexity.
                list_loss.append(raw_loss.item())
                if self.args.variational:
                    if batch_ind % self.args.log_interval == 0 and batch_ind > 0:
                        time_int = time.time() - begin_time
                        loss_int = loss_int_sum.item() / self.args.log_interval
                        loss_int_sum = 0

                    begin_time = time.time()
            cur_loss = total_loss.item() / (batch_ind+1)
            cur_ppl = total_perp / (batch_ind+1)
            acc = corrent / word_num
            accs.append(acc)
            elapsed = time.time() - start_time
            '''print('| client_{} | epoch {:3d} | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}, top3_acc {:.5f}'.format(
                self.user, iter, optimizer.param_groups[0]['lr'], elapsed * 1000 / (batch_ind+1), cur_loss, cur_ppl, corrent / word_num))'''

        val_loss = self.evaluate(data_loader=self.valdata_loader, model=net)
        '''print('| client_{} | val_loss {:5.2f} | val_ppl {:8.2f}, val_top3_acc {:.5f}'.format(self.user, val_loss[0], val_loss[1], val_loss[2]))'''

        params_dicts = net.cpu().state_dict()
        if self.args.variational:
            del params_dicts['rnns.0.module.weight_hh_l0_raw']
            del params_dicts['rnns.1.module.weight_hh_l0_raw']
            del params_dicts['decoder.weight']

        if self.args.dgc:
            # print(f'\n==> initializing dgc compression')
            # self.memory.initialize(params_dicts)
            self.compression.initialize(params_dicts)
            for k in params_dicts.keys():
                params_dicts[k] = self.compression.compress(params_dicts[k], k)
        # print('| client_{} | total time {:5.2f} s|'.format(self.user, time.time() - total_time_s))

        return {'params': params_dicts,
                'loss': sum(list_loss) / len(list_loss),
                'ppl': sum(list_pp) / len(list_pp),
                'acc': sum(accs) / len(accs),
                'val_loss': val_loss}

    def warmup_compress_ratio(self):
        compress_ratio = 0.75
        epoch = self.round
        if epoch > 20 and epoch < 30:
            compress_ratio = 0.9375
        elif epoch >= 30 and epoch < 40:
            compress_ratio = 0.984375
        elif epoch >= 40 and epoch < 50:
            compress_ratio = 0.996
        elif epoch >= 50:
            compress_ratio = 0.999
        return compress_ratio

    def evaluate(self, data_loader, model):
        """ Perplexity of the given data with the given model. """
        model.eval()
        hidden = model.init_hidden(self.args.bs)
        with torch.no_grad():
            word_count = 0
            entropy_sum = 0
            corrent = 0
            loss_list = []
            for val_idx, sents in enumerate(data_loader):
                x = torch.stack(sents[:-1])
                y = torch.stack(sents[1:]).view(-1)
                if self.args.gpu != -1:
                    x, y = x.cuda(), y.cuda()
                    model = model.cuda()
                if hidden[0][0].size(1) != x.size(1):
                    hidden = model.init_hidden(x.size(1))
                    out, hidden = model(x, hidden)
                out, hidden = model(x, hidden)
                loss = self.loss_func(out, y)
                loss_list.append(loss)
                prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
                entropy_sum += prob.log2().neg().sum().item()
                word_count += y.size(0)
                top_3, top3_index = torch.topk(out, 3, dim=1)
                for i in range(top3_index.size(0)):
                    if y.data[i] in top3_index[i]:
                        corrent += 1
            eval_acc = corrent / word_count
            loss_avg = sum(loss_list) / len(loss_list)
        return [loss_avg.item(), 2 ** (entropy_sum / word_count), eval_acc]

