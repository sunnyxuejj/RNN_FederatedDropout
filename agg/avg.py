#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import torch
import numpy as np

def average_weights(w, avg_weight, args):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w[0])
    avg_weight = np.array(avg_weight)
    avg_weight = avg_weight / sum(avg_weight)
    for i in range(len(w)):
        for k in w[i]:
            w[i][k] = w[i][k] * avg_weight[i]
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(w[0][k])
        for i in range(len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = w_avg[k] + torch.mul(torch.randn(w_avg[k].shape), args.dp)
    return w_avg


