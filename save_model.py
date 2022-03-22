#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import sys
import datetime
from time import strftime,gmtime

def save_model_named(args):
    # log 名字设定
    now_time = strftime("%Y-%m-%d %H:%M", gmtime())
    now_time = now_time.replace(' ', '_')
    now_time = now_time.replace(':', '-')
    file_name = args.model_save_path + '/' + \
                args.model +'/'+ \
                args.model+'_'+ args.atten_type + str(now_time) + '.pth'
    print(file_name)
    return file_name


