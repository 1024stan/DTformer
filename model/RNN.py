#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        """
        input_dim: 每个输入xi的维度
        hidden_dim: 词向量嵌入变换的维度，也就是W的行数
        layer_dim: RNN神经元的层数
        output_dim: 最后线性变换后词向量的维度
        """
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim,
            batch_first = True,
            nonlinearity = "relu"
        )

        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        维度说明：
            time_step = sum(像素数) / input_dim
            x : [batch, time_step, input_dim]
        """
        out, h_n = self.rnn(x, None)   # None表示h0会以全0初始化，及初始记忆量为0
        """
        out : [batch, time_step, hidden_dim]
        """
        out = self.fc1(out[: , -1, :])   # 此处的-1说明我们只取RNN最后输出的那个h。
        return out




