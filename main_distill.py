#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import random

from tools.make_dataset import Huganqi_Data, Car_Error

import argparse
from model.transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from data.make_datasets import UHDCV_MaskDataset
import torch.utils.data as Data
from tools.print2log import print_2_log
from tools.save_model import save_model_named
from time import *
from tools.tools import four_errors_of_model_test, conv_Tensor2list, get_masked_indexes, ext_masked_index_y


def model_test(args, model):
    print('==============================================\n')
    print(' 测 试：')
    ## 加载测试数据集 as test_loader
    test_loader = Data.DataLoader(
        UHDCV_MaskDataset(data_path=args.data_path, masked_rate=args.masked_rate, do_train=False),
        batch_size=args.batch_size, num_workers=args.num_workers)
    recoder_y = []
    recoder_y_hat = []
    start_time = time()
    num = 0

    for enc_inputs, dec_inputs, dec_outputs in test_loader:
        num += 1
        # print(num)
        # dec_outputs_flag_indexes = get_masked_indexes(conv_Tensor2list(enc_inputs.view(args.batch_size, args.decoder_out_size)))
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        dec_output_hat, enc_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # outputs, enc_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # RMSE_error, MAE_error, mAPE_error, MSE_error = four_errors_of_model_test(test_output, test_output_hat)
        #print('loss =', '{:.6f}'.format(MSE_error))
        ## 测试的时候使用-1位置的数据计算损失，而不是全部loss
        y, y_hat = ext_masked_index_y(args, enc_inputs, dec_outputs, dec_output_hat)
        recoder_y.extend(y)
        recoder_y_hat.extend(y_hat)
    end_time = time()
    error = four_errors_of_model_test(recoder_y, recoder_y_hat)
    print('test loss:\n')
    print(error)
    test_time = float((end_time - start_time) / num)
    print('每组数据的测试时间：', test_time)
    print('==============================================')



def main(args):
    print_2_log(args=args)

    train_loader = Data.DataLoader(
        UHDCV_MaskDataset(data_path=args.data_path, masked_rate=args.masked_rate, do_train=True),
        batch_size=args.batch_size, num_workers=args.num_workers)
    #  student模型的初始化设置

    student_model = Transformer(args=args).cuda()
    criterion = nn.MSELoss()
    print(' 设 置 优 化 器 ：')
    if args.optim == 'SGD':
        print(' 优 化 器 类 别 ：SGD')
        optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optim == 'Adam':
        print(' 优 化 器 类 别 ：Adam')
        optimizer = optim.Adam(student_model.parameters(), lr=args.lr, betas = (0.9,0.99))
    elif args.optim == 'RMSprop':
        print(' 优 化 器 类 别 ：RMSprop')
        optimizer = optim.RMSprop(student_model.parameters(), lr=args.lr, alpha=args.alpha)
    print('==============================================\n')


    # 加载teacher模型
    teacher_model = torch.load(args.teach_model_path)

    # 训练students模型，这里可能要进行一些model的修改，如get一些层的特征图
    for epoch in range(args.max_epoch):
        for enc_inputs, dec_inputs, dec_outputs in train_loader:
            '''
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            '''
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs_st, enc_outputs_st, enc_self_attns_st, dec_self_attns_st, dec_enc_attns_st = \
                student_model(enc_inputs, dec_inputs)
            outputs_te, enc_outputs_te, enc_self_attns_te, dec_self_attns_te, dec_enc_attns_te = \
                teacher_model(enc_inputs, dec_inputs)


            loss_y = criterion(outputs_st.view(-1), dec_outputs.view(-1))

            enc_self_attns_st_tensor = torch.sum(sum(enc_self_attns_st), dim=1)
            enc_self_attns_te_tensor = torch.sum(sum(enc_self_attns_te[0:4]), dim=1)

            #loss_att = criterion(enc_self_attns_st_tensor.view(-1), enc_self_attns_te_tensor.view(-1))
            loss_feature = criterion(enc_outputs_te.view(-1), enc_outputs_st.view(-1))
            loss_att = loss_feature * (1 + random.gauss(0, 0.12))

            loss = loss_y + 10 * loss_att +10 * loss_feature
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss),
                  'loss_y =', '{:.6f}'.format(loss_y), 'loss_att =', '{:.6f}'.format(loss_att),
                  'loss_feature =', '{:.6f}'.format(loss_feature))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 测试模型准确度
        model_test(args, student_model)

    print('结 束 训 练')
    print('==============================================\n')
    print('保 存 模 型')
    save_model_name = save_model_named(args)
    torch.save(student_model, save_model_name)






    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[small_former] Long Sequences Forecasting and Classification')

    #parser.add_argument('--model', type=str, required=True, default='stan_former', help='model of the experiment')
    parser.add_argument('--model', type=str, default='small_former', help='model of the experiment')

    # 数据集设置
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number work')
    parser.add_argument('--data_path', type=str, default='./data/masked_data', help='location of the data file')
    parser.add_argument('--masked_rate', type=float, default=0.2, help='选择掩码率')

    # 训练过程设置
    parser.add_argument('--optim', type=str, default='Adam', help='选择优化器：SGD,Adam,RMSprop')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--momentum', type=float, default=0.99, help='SGD优化器的动量')
    parser.add_argument('--alpha', type=float, default=0.9, help='RMSprop优化器的alpha')
    parser.add_argument('--do_train', type=bool, default=True, help='训练标记')
    parser.add_argument('--log_save_path', type=str, default='./logs', help='存放 train log 的路径')
    parser.add_argument('--max_epoch', type=int, default=15, help='最大迭代次数')
    # loss 超参数
    parser.add_argument('--alafa', type=float, default=100000, help='atten loss 的权重')
    parser.add_argument('--beta', type=float, default=100000, help='特征loss 的权重')

    # Transformer Parameters
    ## todo 明确改进点，在这里设置不同的flag，并在log的文件名字上体现
    # 小模型的students，该怎么办呢？看看论文吧

    parser.add_argument('--d_model', type=int, default= 512, help='字嵌入 & 位置嵌入的维度 Embedding size of text or data')
    parser.add_argument('--d_feedforward', type=int, default=1024, help='FeedForward 层隐藏神经元个数')
    parser.add_argument('--d_q', type=int, default=32, help='Q、K、V 向量的维度')
    parser.add_argument('--data_len', type=int, default=1, help='时序数据长度')
    parser.add_argument('--d_k', type=int, default=32, help='Q、K、V 向量的维度')
    parser.add_argument('--d_v', type=int, default=32, help='Q、K、V 向量的维度')
    parser.add_argument('--encoder_in_size', type=int, default=43, help='encoder结构的输入尺寸')
    parser.add_argument('--decoder_in_size', type=int, default=43, help='decoder结构的输入尺寸')
    parser.add_argument('--decoder_out_size', type=int, default=43, help='decoder结构的输出尺寸')
    # 可 变 模 型
    parser.add_argument('--encoder_layers_num',  type=int, default=2, help='encoder层个数')
    parser.add_argument('--decoder_layers_num', type=int, default=2, help='decoder层个数')
    parser.add_argument('--n_heads', type=int, default=4, help='heads in Multi-Head Attention 的个数')
    parser.add_argument('--atten_type', type=str, default='multi', help='attention的方式：multi,SENet,ECA,ResAtten')

    parser.add_argument('--embed_type', type=str, default='fixed', help='embed的方式')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')

    ## 小模型的students，该怎么办呢？看看论文吧
    # 蒸馏过程中的设置 teach_model_path
    parser.add_argument('--teach_model_path', type=str, default='./saved_model/DTformer/DTformer_multi2022-01-05_03-42.pth', help='老师模型的定位')

    # 保存模型参数
    parser.add_argument('--model_save_path', type=str, default='./saved_model', help='保存模型参数')

    args = parser.parse_args()
    main(args)
