#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import numpy as np

def partition(len_dataset, num_users):
    num_items = int(len_dataset/num_users)
    dict_users, all_idxs = {}, [i for i in range(len_dataset)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def data_patition(train_data_total, val_data_total, test_data_total, client_num):
    train_data = data_split(train_data_total, client_num)
    val_data = data_split(val_data_total, client_num)
    test_data = data_split(test_data_total, client_num)
    return train_data, val_data, test_data

def data_split(data, num):
    data_size = data.size(0) // num
    result = data.narrow(0, 0, num * data_size)
    result = result.view(data_size, -1).t().contiguous()
    return result