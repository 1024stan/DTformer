#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import time
import torch
from torch.nn import Module, LSTM, Linear, Conv1d
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tools import *
from time import *
import torch.nn as nn


class BiLSTM(Module):
    def __init__(self, input_dim, output_dim,use_bidirectional,
                 hidden_size,num_layers_lstm, dropout_rate):
        super(BiLSTM, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True,
        #                  dropout=config.dropout_rate, bidirectional=False)
        if use_bidirectional == True:
            lstm2liner_features = hidden_size * 2
        elif use_bidirectional == False:
            lstm2liner_features = hidden_size

        self.lstm = LSTM(input_size=input_dim, hidden_size=hidden_size,
                         num_layers=num_layers_lstm, batch_first=True,
                         dropout=dropout_rate, bidirectional=use_bidirectional)
        self.linear_1 = Linear(in_features=lstm2liner_features, out_features=input_dim)
        self.linear_2 = Linear(in_features=input_dim, out_features=output_dim)


    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_1_out = self.linear_1(lstm_out)
        # linear_1_out = linear_1_out.view(linear_1_out.shape[0], linear_1_out.shape[2], linear_1_out.shape[1])
        linear_out = self.linear_2(linear_1_out)
        return linear_out

