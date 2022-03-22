#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import argparse
from model.losses import RateLoss
from model.transformer import Transformer
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as Dataloader
from data.make_datasets import UHDCV_RegularDataset
import torch.utils.data as Data
from tools.print2log import print_2_log
from model.RNN import RNN
from model.ResNet import ResNet
from model.BiLSTM import BiLSTM
import time
from time import *
from tools.tools import four_errors_of_model_test


def main(args):
    print_2_log(args=args)

    print(' 加 载 训 练 集 ：')
    train_loader = Data.DataLoader(
        UHDCV_RegularDataset(data_path=args.data_path+'/train.txt'), batch_size=args.batch_size, num_workers=args.num_workers)
    print('==============================================\n')

    ## 随机划分数据集为训练集和测试集
    # train_size = int(0.8 * len(loader))
    # test_size = len(loader) - train_size
    # train_dataset, test_dataset = Data.random_split(Huganqi_Data, [train_size, test_size])


    print(' 加 载 模 型 ：')
    if args.model == 'RNN':
        print(' 模 型 类 别 ：RNN')
        model = RNN(input_dim=args.input_dim, hidden_dim=args.rnn_hidden_dim,
                    layer_dim=args.rnn_layer_dim,output_dim=args.output_dim).cuda()
    elif args.model == 'CNN':
        print(' 模 型 类 别 ：CNN')
        model = ResNet(input_dim=args.input_dim, output_dim=args.output_dim,
                       hidden_dim_list=args.cnn_hidden_dim_list, num_blocks_in_layer=args.cnn_num_blocks_in_layer).cuda()
    elif args.model == 'BiLSTM':
        print(' 模 型 类 别 ：BiLSTM')
        model = BiLSTM(input_dim=args.input_dim, output_dim=args.output_dim,use_bidirectional=args.use_bidirectional,
                       hidden_size=args.lstm_hidden_size, num_layers_lstm=args.num_layers_lstm, dropout_rate=args.dropout_rate_lstm).cuda()
    print('==============================================\n')

    criterion = RateLoss()
    print(' 设 置 优 化 器 ：')
    if args.optim == 'SGD':
        print(' 优 化 器 类 别 ：SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'Adam':
        print(' 优 化 器 类 别 ：Adam')
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas = (0.9,0.99))
    elif args.optim == 'RMSprop':
        print(' 优 化 器 类 别 ：RMSprop')
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    print('==============================================\n')


    print(' 开 始 训 练 ：')
    test_loader = train_loader

    for epoch in range(args.max_epoch):
        for inputs, outputs in train_loader:
            '''
            inputs: [batch_size, src_len]
            outputs: [batch_size, tgt_len]
            '''
            input, output = inputs.cuda(), outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            output_hat = model(input)
            loss = criterion(output, output_hat)
            if loss > 0:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            # print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss), 'y:',list(output), 'y_hat:',list(output_hat))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                print('error: y:',list(output), 'y_hat:',list(output_hat))

        print('==============================================\n')

        print(' 测 试：')
        recoder_y = []
        recoder_y_hat = []
        start_time = time()
        num = 0
        for test_input, test_output in test_loader:
            num += 1
            # print('\r',num,end='')
            test_input, test_output = test_input.cuda(), test_output.cuda()
            test_output_hat = model(test_input)
            # RMSE_error, MAE_error, mAPE_error, MSE_error = four_errors_of_model_test(test_output, test_output_hat)
            #print('loss =', '{:.6f}'.format(MSE_error))
            recoder_y.extend(test_output)
            recoder_y_hat.extend(test_output_hat)
        end_time = time()
        error = four_errors_of_model_test(recoder_y, recoder_y_hat)
        print('test loss:\n')
        print(error)

    print('结 束 训 练')
    print('==============================================\n')


    print(' 加 载 测 试 集 ：')
    #test_loader = train_loader
    test_loader = Data.DataLoader(UHDCV_RegularDataset(data_path=args.data_path+'/test.txt'), batch_size=args.batch_size, num_workers=args.num_workers)

    print('==============================================\n')

    print(' 测 试：')
    recoder_y = []
    recoder_y_hat = []
    start_time = time()
    num = 0
    for test_input, test_output in test_loader:
        num += 1
        # print('\r',num,end='')
        test_input, test_output = test_input.cuda(), test_output.cuda()
        test_output_hat = model(test_input)
        # RMSE_error, MAE_error, mAPE_error, MSE_error = four_errors_of_model_test(test_output, test_output_hat)
        #print('loss =', '{:.6f}'.format(MSE_error))
        recoder_y.extend(test_output)
        recoder_y_hat.extend(test_output_hat)
    end_time = time()
    error = four_errors_of_model_test(recoder_y, recoder_y_hat)
    print('test loss:\n')
    print(error)
    test_time = float((end_time - start_time) / num)
    print('每组数据的测试时间：', test_time)
    print('==============================================')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[stan_regular] CNN RNN LSTM model to c')

    #parser.add_argument('--model', type=str, required=True, default='stan_former', help='model of the experiment')
    parser.add_argument('--model', type=str, default='BiLSTM', help='model of the experiment：CNN, RNN, BiLSTM')
    parser.add_argument('--log_save_path', type=str, default='./logs', help='存放 train log 的路径')
    parser.add_argument('--data_path', type=str, default='./data/UHDCV_DTdataset', help='存放 train文件的路径')

    # 通用模型的设置
    parser.add_argument('--input_dim', type=int, default=36, help='输入每个xi的维度')
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度，即线损的程度')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2, help='num of work')




    # 训练方式设置
    parser.add_argument('--optim', type=str, default='Adam', help='选择优化器：SGD,Adam,RMSprop')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.99, help='SGD优化器的动量')
    parser.add_argument('--alpha', type=float, default=0.9, help='RMSprop优化器的alpha')
    parser.add_argument('--max_epoch', type=int, default=40, help='epoch次数')


    # RNN模型的设置参数
    parser.add_argument('--rnn_hidden_dim', type=int, default=512, help='词向量嵌入变换的维度，也就是隐藏层')
    parser.add_argument('--rnn_layer_dim', type=int, default=4, help='RNN神经元的层数')

    # CNN模型的设置参数
    parser.add_argument('--cnn_hidden_dim_list', type=list, default=[64, 128, 256, 512], help='resnet中每一层的通道数')
    parser.add_argument('--cnn_num_blocks_in_layer', type=int, default=1, help='resnet中每一层网络的block数')


    # BiLSTM模型的设置参数
    parser.add_argument('--use_bidirectional', type=bool, default=True, help='是否使用双向结构')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='lstm的隐藏层特征尺度')
    parser.add_argument('--num_layers_lstm', type=int, default=8, help='lstm的层数')
    parser.add_argument('--dropout_rate_lstm', type=float, default=0.1, help='resnet中每一层网络的block数')



    args = parser.parse_args()
    main(args)




