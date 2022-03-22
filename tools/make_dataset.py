#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import math
import logging
from logging.handlers import RotatingFileHandler
import sys
from torch.utils.data import Dataset


class UHDCV_Data(Dataset):
    def __init__(self, args=None):
        super(UHDCV_Data, self).__init__()
        self.args = args
        # todo 设定特高压直流输电数据集的参数
        self.jiao_or_bi = 'bicha'
        self.valid_data_rate = valid_data_rate
        self.file_name = './data/huganqi/all_change.xlsx'
        self.x_data, self.bicha_data, self.jiaocha_data = self.read_data()
        self.enc_inputs, self.dec_inputs, self.dec_outputs = self.get_train_and_valid_data()


    def read_data(self):                # 读取初始数据

        # todo 读取原始数据，在这之前最好存好要读的东西，确定好 x_data 和 y_data 的格式

        df = pd.read_excel(self.file_name)
        df.dropna(inplace=True)
        data_in_file = df.values
        x_data = data_in_file[:, 0:-2]
        bicha_data = data_in_file[:, -2].reshape(-1, 1)
        jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data)) / self.args.history_times) - self.args.history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + self.args.history_times, :])
            bicha_y.append(bicha_data[i + self.args.history_times])
            jiaocha_y.append(jiaocha_data[i + self.args.history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y).reshape([max_length, 1, 1])
        jiaocha_y = np.array(jiaocha_y).reshape([max_length, 1, 1])
        return mndata, bicha_y, jiaocha_y

    def get_train_and_valid_data(self):
        if self.jiao_or_bi == 'bicha':
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.bicha_data, test_size=self.valid_data_rate,
                                                                  random_state=self.args.random_seed)
        elif self.jiao_or_bi ==  'jiaocha':
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.jiaocha_data,
                                                                  test_size=self.args.valid_data_rate,
                                                                  random_state=self.args.random_seed,
                                                                  shuffle=self.args.shuffle_train_data)
        else:
            print("请选择计算角差还是计算比差")

        if self.args.do_train == True:  # 划分训练和验证集，并打乱
            return torch.FloatTensor(train_x), torch.FloatTensor(train_x), torch.FloatTensor(train_y)
        else:
            return torch.FloatTensor(valid_x), torch.FloatTensor(valid_x), torch.FloatTensor(valid_y)

    def get_all_data(self):
        if self.jiao_or_bi == 'bicha':
            return self.x_data, self.bicha_data
        elif self.jiao_or_bi == 'jiaocha':
            return self.x_data, self.jiaocha_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, item):
        return self.enc_inputs[item], self.dec_inputs[item], self.dec_outputs[item]




class Car_Error(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, *args, **kwargs):
        pass




class Huganqi_Data(Dataset):
    def __init__(self, args=None, jiao_or_bi=None, valid_data_rate=0.3):
        super(Huganqi_Data, self).__init__()
        self.args = args
        self.jiao_or_bi = 'bicha'
        self.valid_data_rate = valid_data_rate
        self.file_name = './data/huganqi/all_change.xlsx'
        self.x_data, self.bicha_data, self.jiaocha_data = self.read_data()
        self.enc_inputs, self.dec_inputs, self.dec_outputs = self.get_train_and_valid_data()


    def read_data(self):                # 读取初始数据
        df = pd.read_excel(self.file_name)
        df.dropna(inplace=True)
        data_in_file = df.values
        x_data = data_in_file[:, 0:-2]
        bicha_data = data_in_file[:, -2].reshape(-1, 1)
        jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data)) / self.args.history_times) - self.args.history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + self.args.history_times, :])
            bicha_y.append(bicha_data[i + self.args.history_times])
            jiaocha_y.append(jiaocha_data[i + self.args.history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y).reshape([max_length, 1, 1])
        jiaocha_y = np.array(jiaocha_y).reshape([max_length, 1, 1])
        return mndata, bicha_y, jiaocha_y

    def get_train_and_valid_data(self):
        if self.jiao_or_bi == 'bicha':
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.bicha_data, test_size=self.valid_data_rate,
                                                              random_state=self.args.random_seed)
        elif self.jiao_or_bi ==  'jiaocha':
            train_x, valid_x, train_y, valid_y = train_test_split(self.x_data, self.jiaocha_data,
                                                                  test_size=self.args.valid_data_rate,
                                                                  random_state=self.args.random_seed,
                                                                  shuffle=self.args.shuffle_train_data)
        else:
            print("请选择计算角差还是计算比差")

        if self.args.do_train == True:  # 划分训练和验证集，并打乱
            return torch.FloatTensor(train_x), torch.FloatTensor(train_x), torch.FloatTensor(train_y)
        else:
            return torch.FloatTensor(valid_x), torch.FloatTensor(valid_x), torch.FloatTensor(valid_y)

    def get_all_data(self):
        if self.jiao_or_bi == 'bicha':
            return self.x_data, self.bicha_data
        elif self.jiao_or_bi == 'jiaocha':
            return self.x_data, self.jiaocha_data

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, item):
        return self.enc_inputs[item], self.dec_inputs[item], self.dec_outputs[item]






def read_excel_file(file_name=None, sheet_name=None, flag='nto1',history_times=10):
    df = pd.read_excel(file_name, sheet_name)
    df.dropna(inplace=True)
    data_in_file = df.values
    x_data = data_in_file[:,0:-2]
    bicha_data = data_in_file[:, -2].reshape(-1, 1)
    jiaocha_data = data_in_file[:, -1].reshape(-1, 1)
    if flag == 'nto1':
        return x_data, bicha_data, jiaocha_data
    elif flag == 'mnto1':

        mn_x_data = []
        bicha_y = []
        jiaocha_y = []
        max_length = int(len(list(bicha_data))/history_times)- history_times
        for i in range(max_length):
            mn_x_data.append(x_data[i: i + history_times, :].reshape(-1))
            bicha_y.append(bicha_data[i + history_times])
            jiaocha_y.append(jiaocha_data[i + history_times])
        mndata = np.array(mn_x_data)
        bicha_y = np.array(bicha_y)
        jiaocha_y = np.array(jiaocha_y)
        return mndata, bicha_y, jiaocha_y



def save_data_as_csv(file_name, save_data, string=''):

    save_temp_data = pd.DataFrame(data=save_data)
    work_name = file_name.split('/')[2].split('.')[0]
    temp = np.mat(save_temp_data)
    save_temp_data_name = './data/' + work_name + '_' + string + '.csv'
    save_temp_data.to_csv(save_temp_data_name, encoding='gbk')

def split_dataset(train_data=None, train_target=None):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.4,
                                                                         random_state=0)
    return X_train, X_test, y_train, y_test

def lei_pso_decay_coefficient_mean(all_particle_xx=None, best_xx=None,flag=None):
    mean_value = 0
    distance = []
    if flag == "D":
        for i in range(len(all_particle_xx)):
            distance.append(math.sqrt((all_particle_xx[i][0] - best_xx[0])**2 + (all_particle_xx[i][1] - best_xx[1])**2))
            mean_value = mean_value + distance[i]
    elif flag == "F":
        for i in range(len(all_particle_xx)):
            distance.append(all_particle_xx[i] - best_xx)
            mean_value = mean_value + distance[i]
    mean_value = mean_value / len(all_particle_xx)
    return mean_value, distance







