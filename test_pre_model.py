#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
from model.transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as Dataloader
from data.make_datasets import UHDCV_MaskDataset
#from tools.make_dataset import Huganqi_Data, Car_Error
from tools.load_logger import load_logger
import torch.utils.data as Data
from tools.print2log import print_2_log
from tools.save_model import save_model_named
from time import *
from tools.tools import four_errors_of_model_test, conv_Tensor2list, get_masked_indexes, ext_masked_index_y


def model_test(args):
    print_2_log(args=args)
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

    model = torch.load(args.test_model_path)

    for enc_inputs, dec_inputs, dec_outputs in test_loader:
        num += 1
        print(num)
        # dec_outputs_flag_indexes = get_masked_indexes(conv_Tensor2list(enc_inputs.view(args.batch_size, args.decoder_out_size)))
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
        dec_output_hat, enc_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # outputs, enc_outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # RMSE_error, MAE_error, mAPE_error, MSE_error = four_errors_of_model_test(test_output, test_output_hat)
        #print('loss =', '{:.6f}'.format(MSE_error))
        ## todo 测试的时候使用-1位置的数据计算损失，而不是全部loss
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






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[stan_former] Long Sequences Forecasting and Classification')
    #parser.add_argument('--model', type=str, required=True, default='stan_former', help='model of the experiment')
    parser.add_argument('--model', type=str, default='DTformer-test', help='model of the experiment')
    # 数据集设置
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
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
    parser.add_argument('--max_epoch', type=int, default=1, help='最大迭代次数')


    # 保存模型参数
    parser.add_argument('--model_save_path', type=str, default='./saved_model', help='dropout')
    parser.add_argument('--test_model_path', type=str, default='./saved_model/DTformer/DTformer_multi2021-12-26_06-12.pth', help='测试模型的路径')

    args = parser.parse_args()

    model_test(args)
