#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # temp = torch.sqrt(x - y)
        # temp = torch.mean(temp)

        return torch.mean(torch.sqrt(torch.abs(x - y))) + 1

def conv_Tensor2list(Tensor):
    Tensor = Tensor.cpu()
    Tensor = Tensor.numpy()
    Tensor = Tensor.tolist()
    return Tensor

def get_masked_indexes(data_lists):
    indexes = []
    index = []
    for data in data_lists:
        for i, value in enumerate(data):
            if value == -1:
                index += [i]
        indexes.append(index)
        index = []

    return indexes