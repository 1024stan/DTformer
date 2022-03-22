#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import logging
from logging.handlers import RotatingFileHandler
import sys
import time

def load_logger(args):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    start_time = time

    # StreamHandler
    if args.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if args.do_log_save_to_file:
        file_handler = RotatingFileHandler(args.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(args):
            if not key.startswith("_"):
                config_dict[key] = getattr(args, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger