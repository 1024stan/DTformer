#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import sys
import datetime
from time import strftime,gmtime
import sys
import os

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# path = os.path.abspath(os.path.dirname(__file__))	#获取当前py文件的父目录
# type = sys.getfilesystemencoding()
# sys.stdout = Logger('a.txt')						#输出文件

def print_2_log(args):
    # log 名字设定
    now_time = strftime("%Y-%m-%d %H:%M", gmtime())
    now_time = now_time.replace(' ', '_')
    now_time = now_time.replace(':', '-')
    file_name = args.log_save_path + '/' + args.model + str(now_time) + '.log'
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(file_name)
    # 输出设置内容
    config_dict = {}
    for key in dir(args):
        if not key.startswith("_"):
            config_dict[key] = getattr(args, key)
    config_str = str(config_dict)
    config_list = config_str[1:-1].split(", '")
    print('==============================================')
    config_save_str = "Config:\n" + "\n'".join(config_list)
    print(config_save_str)
    print('==============================================\n')

    # 接下来自己发挥



'''

def print_2_log(args):
    # log 名字设定
    now_time = strftime("%Y-%m-%d %H:%M", gmtime())
    now_time = now_time.replace(' ', '_')
    now_time = now_time.replace(':', '-')
    file_name = args.log_save_path + '/' + args.model + str(now_time) + '.log'
    sys.stdout = open(file=file_name, mode='w', encoding='utf-8')
    # 输出设置内容
    config_dict = {}
    for key in dir(args):
        if not key.startswith("_"):
            config_dict[key] = getattr(args, key)
    config_str = str(config_dict)
    config_list = config_str[1:-1].split(", '")
    print('==============================================')
    config_save_str = "Config:\n" + "\n'".join(config_list)
    print(config_save_str)
    print('==============================================\n')

    # 接下来自己发挥

'''