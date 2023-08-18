#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.8
import math
import random

import numpy as np
import torch
import torch.nn as nn
import pickle

from torch.utils.data import DataLoader
from dgc.memory import DGCSGDMemory
from Update import LocalUpdateLM
from agg.avg import *
from Text import DatasetLM
from utils.options import args_parser
from Models import RNNModel
from data.user_data import data_process

args = args_parser()


def evaluate(data_loader, model):
    """ Perplexity of the given data with the given model. """
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    hidden = model.init_hidden(args.bs)
    with torch.no_grad():
        word_count = 0
        entropy_sum = 0
        corrent = 0
        loss_list = []
        for val_idx, sents in enumerate(data_loader):
            x = torch.stack(sents[:-1])
            y = torch.stack(sents[1:]).view(-1)
            if args.gpu != -1:
                x, y = x.cuda(), y.cuda()
                model = model.cuda()
            if hidden[0][0].size(1) != x.size(1):
                hidden = model.init_hidden(x.size(1))
                out, hidden = model(x, hidden)
            out, hidden = model(x, hidden)
            loss = loss_func(out, y)
            loss_list.append(loss)
            prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.size(0)
            top_3, top3_index = torch.topk(out, 3, dim=1)  # top3 accuracy for next-word prediction
            for i in range(top3_index.size(0)):
                if y.data[i] in top3_index[i]:
                    corrent += 1
        eval_acc = corrent / word_count
        loss_avg = sum(loss_list) / len(loss_list)
    return [loss_avg.item(), 2 ** (entropy_sum / word_count), eval_acc]


if __name__ == "__main__":
    data_dir = './data/train/'

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open('./data/reddit_vocab.pck', 'rb') as f:
        vocab = pickle.load(f)
    config = args
    nvocab = vocab['size']
    train_data, val_data, test_data = data_process(data_dir, nvocab, args.nusers)
    dataset_train, dataset_val, data_num = {}, {}, {}
    for i in range(args.nusers):
        dataset_train[i] = DatasetLM(train_data[i], vocab['vocab'])
        dataset_val[i] = DatasetLM(val_data[i], vocab['vocab'])
        data_num[i] = len(train_data[i])

    lr = args.lr
    best_val_loss = None
    model_saved = '../log/model_avg.pt'
    loss_train = []
    memory = {}

    net_glob = RNNModel(config, nvocab)
    if args.gpu != -1:
        net_glob = net_glob.cuda()
    w_glob = net_glob.cpu().state_dict()
    val_ppl = {}
    for i in range(args.nusers):
        val_ppl[i] = 10e9
        memory[i] = DGCSGDMemory()
        memory[i].initialize(w_glob)
    before_loss = 0
    before_val_acc = None
    acc_red_num = 0
    sd_flag = False

    try:
        for epoch in range(args.epochs):
            w_locals, loss_locals, pp_locals, acc_locals, mask_locals, avg_weight = [], [], [], [], [], []
            m = max(int(args.frac * args.nusers), 1)
            idxs_users = np.random.choice(range(args.nusers), m, replace=False)  # Randomly sample clients
            val_loss_list, val_loss = [], {}
            total_num = 0
            for idx in idxs_users:
                local = LocalUpdateLM(args=args, train_dataset=dataset_train[idx], val_dataset=dataset_val[idx],
                                      nround=epoch, user=idx, data_num=data_num[idx], memory=memory[idx])
                net_glob.load_state_dict(w_glob)
                out_dict = local.update_weights(net=copy.deepcopy(net_glob), lr=lr, current_epoch=epoch)
                if args.dgc:
                    for k in out_dict['params'].keys():
                        out_dict['params'][k] = local.compression.decompress(out_dict['params'][k][0],
                                                                             out_dict['params'][k][1])
                w_locals.append(copy.deepcopy(out_dict['params']))
                avg_weight.append(data_num[idx])
                loss_locals.append(copy.deepcopy(out_dict['loss']))
                pp_locals.append(copy.deepcopy(out_dict['ppl']))
                acc_locals.append(copy.deepcopy(out_dict['acc']))
                val_loss[idx] = out_dict['val_loss']
                val_loss_list.append(np.array(out_dict['val_loss']) * data_num[idx])
                total_num += data_num[idx]

            # update global weights
            loss_avg = sum(loss_locals) / len(loss_locals)
            acc_avg = sum(acc_locals) / len(acc_locals)
            ppl_avg = sum(pp_locals) / len(pp_locals)
            print('\nTrain loss: {:.5f}, train ppl: {:.5f}, train top3_acc: {:.5f}'.format(loss_avg, ppl_avg, acc_avg), flush=True)
            loss_train.append(loss_avg)

            val_loss_avg = np.sum(np.array(val_loss_list), axis=0) / total_num
            print("Epoch {}, Validation loss: {:.5f}, val ppl: {:.5f}, val top3_acc: {:.5f}".format(epoch, val_loss_avg[0], val_loss_avg[1], val_loss_avg[2]), flush=True)

            if args.agg == 'avg':
                w_glob = average_weights(w_locals, avg_weight, args)
            elif args.agg == 'att':
                w_glob = aggregate_att(w_locals, w_glob, args)
            elif args.agg == 'med':
                if abs(before_loss - loss_avg) > 0.01:
                    w_glob = adaptive_agg(w_locals, w_glob, args.epsilon, dp=args.dp)
                else:
                    w_glob = average_weights(w_locals, avg_weight, args)
            elif args.agg == 'SD_att':
                if sd_flag:
                    w_glob = SD_att_agg(w_locals, w_glob, args.epsilon, args.ord, dp=args.dp)
                else:
                    w_glob = average_weights(w_locals, avg_weight, args)
            else:
                exit('Unrecognized aggregation')
            # copy weight to net_glob
            if args.variational:
                w_glob['decoder.weight'] = w_glob['encoder.weight']
                w_glob['rnns.0.module.weight_hh_l0_raw'] = w_glob['rnns.0.module.weight_hh_l0']
                w_glob['rnns.1.module.weight_hh_l0_raw'] = w_glob['rnns.1.module.weight_hh_l0']
            net_glob.load_state_dict(w_glob)

            if not best_val_loss or val_loss_avg[2] > best_val_loss:
                if best_val_loss:
                    sd_flag = True
                else:
                    sd_flag = False
                with open(model_saved, 'wb') as f:
                    acc_red_num = 0
                    print('save model', flush=True)
                    torch.save(net_glob, f)
                best_val_loss = val_loss_avg[2]
            else:
                acc_red_num += 1
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                if acc_red_num > 6:
                    acc_red_num = 0
                    if lr > 0.001:
                        lr = lr / 4
                    else:
                        lr = 0.001
                    '''args.dropout_hh += 0.1
                    acc_red_num = 0
                    if args.dropout_hh > 0.8:
                        args.dropout_hh = 0.8'''
            before_loss = loss_avg
            before_val_acc = val_loss_avg[2]
            

    except KeyboardInterrupt:
        print('-' * 89)
        print('Existing from training early')

    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)
    
    all_train, all_val, all_test = [], [], []
    for value in train_data.values():
        for sent in value:
            all_train.append(sent)
    for value in val_data.values():
        for sent in value:
            all_val.append(sent)
    for value in test_data.values():
        for sent in value:
            all_test.append(sent)
    dataset_train = DatasetLM(all_train, vocab['vocab'])
    loader_train = DataLoader(dataset=dataset_train, batch_size=args.bs, shuffle=True)
    dataset_val = DatasetLM(all_val, vocab['vocab'])
    loader_val = DataLoader(dataset=dataset_val, batch_size=args.bs, shuffle=True)
    dataset_test = DatasetLM(all_test, vocab['vocab'])
    loader_test = DataLoader(dataset=dataset_test, batch_size=args.bs, shuffle=True)

    pp_train = evaluate(data_loader=loader_train, model=model_best)
    pp_val = evaluate(data_loader=loader_val, model=model_best)
    pp_test = evaluate(data_loader=loader_test, model=model_best)

    print("Train loss: {:.5f}, train ppl: {:.5f}, train top3_acc: {:.5f}".format(pp_train[0], pp_train[1], pp_train[2]))
    print("val loss: {:.5f}, val ppl: {:.5f}, val top3_acc: {:.5f}".format(pp_val[0], pp_val[1], pp_val[2]))
    print("test loss: {:.5f}, test ppl: {:.5f}, test top3_acc: {:.5f}".format(pp_test[0], pp_test[1], pp_test[2]))
