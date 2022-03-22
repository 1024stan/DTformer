#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import copy
import os.path
import random
import numpy as np

def random_list(start,stop,length):
    if length>=0:
        length=int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def save_database_as_txt(filename, data):
    with open(filename, "w") as f:
        for line in data:
            x = str(line[0])
            y = str(line[1])
            xy = x + '=' + y
            xy = xy.replace('[','')
            xy = xy.replace(']','')
            f.write(xy+'\n')

    print('保存到',filename,'中')


def main(args):

    # 读原始数据，以txt的形式
    original_database = []
    with open(args.data_load_path, "r") as f:
        for line in f.readlines():
            line = list(map(float, line.split(',')))
            original_database.append(line)
            # print(line)

    # 增添掩码
    mask_database = copy.deepcopy(original_database)
    masked_database = []
    max_len = len(mask_database[0])
    masked_number = int(max_len * args.mask_rate)
    masked_symbol = args.replaced_mask
    for epoch in range(args.mask_epochs):
        for i, mask_data in enumerate(mask_database):
            mask_list = random_list(0, max_len-1, masked_number)
            # 替换
            for location in mask_list:
                mask_data[location] = masked_symbol
            # 合并 x 和 y
            masked_data = [mask_data, original_database[i]]
            masked_database.append(masked_data)

    # 随机划分训练集和测试集

    train_size = int(len(masked_database) * args.split_rate)
    test_size = len(masked_database) - train_size
    random.shuffle(masked_database)
    train_masked_database = masked_database[0:train_size]
    test_masked_database = masked_database[train_size:-1]

    # 保存训练集和测试集到路径中

    train_masked_database_txt_name = args.masked_data_save_path + '/'+ str(args.mask_rate) + '/train.txt'
    test_masked_database_txt_name = args.masked_data_save_path + '/'+ str(args.mask_rate) + '/test.txt'
    all_masked_database_txt_name = args.masked_data_save_path + '/'+ str(args.mask_rate) + '/all.txt'


    save_database_as_txt(train_masked_database_txt_name, train_masked_database)
    save_database_as_txt(test_masked_database_txt_name, test_masked_database)
    save_database_as_txt(all_masked_database_txt_name, masked_database)
    print('数据集掩码及划分结束')










if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='[stan_regular] CNN RNN LSTM model to c')

    parser.add_argument('--data_load_path', type=str, default='./data/DTdataset-WithLoss.txt', help='存放原始data的路径和文件')
    parser.add_argument('--mask_rate', type=str, default=0.4, help='添加掩码的比例')
    parser.add_argument('--mask_epochs', type=str, default=2, help='添加掩码的epoch')
    parser.add_argument('--split_rate', type=float, default=0.7, help='训练集和测试集划分比例')
    parser.add_argument('--replaced_mask', type=int, default=-1, help='掩码标记')
    parser.add_argument('--masked_data_save_path', type=str, default='./data/masked_data', help='存放掩码后数据的路径')

    args = parser.parse_args()
    main(args)



