#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import math
import logging
from logging.handlers import RotatingFileHandler
import sys
from torch.utils.data import Dataset

class UHDCV_SequenceDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.x, self.y = self.read_data()

    def read_data(self):
        x = []
        y = []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = line.split('=')
                x.append(list(map(float, line[0].split(','))))
                y.append(list(map(float, line[1].split(','))))
        print("数据读取完毕")
        return torch.FloatTensor(x).view(len(x),1,-1), torch.FloatTensor(y).view(len(y), 1, -1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.x[item], self.y[item]



class UHDCV_MaskDataset(Dataset):
    def __init__(self, data_path, masked_rate, do_train):
        self.data_path = data_path
        if do_train:
            self.data_filepath = self.data_path + '/' + str(masked_rate) + '/train.txt'
        else:
            self.data_filepath = self.data_path + '/' + str(masked_rate) + '/test.txt'
        self.x, self.y = self.read_data()

    def read_data(self):
        x = []
        y = []
        with open(self.data_filepath, 'r') as f:
            for line in f.readlines():
                line = line.split('=')
                x.append(list(map(float, line[0].split(','))))
                y.append(list(map(float, line[1].split(','))))
        print("数据读取完毕")
        return torch.FloatTensor(x).view(len(x),1,-1), torch.FloatTensor(y).view(len(y), 1, -1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.x[item], self.y[item]




class UHDCV_RegularDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.x, self.y = self.read_data()

    def read_data(self):
        x = []
        y = []
        with open(self.data_path, 'r') as f:
            for line in f.readlines():
                line = list(map(float, line.split(',')))
                x.append(line[0:36])
                y.append(line[-1])
        print("数据读取完毕")
        return torch.FloatTensor(x).view(len(x),1,-1), torch.FloatTensor(y).view(len(y), 1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], self.y[item]



