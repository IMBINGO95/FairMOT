#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import time
from colorlog import ColoredFormatter

import os


class Log(object):
    '''
封装后的logging
    '''

    def __init__(self, logger=None, log_cate='search'):
        '''
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        '''
        #black, red, green, yellow, blue, purple, cyan and white
        self.CYAN = 21  # INFO level is 20
        self.GREEN = 22
        self.BLUE = 23
        self.PURPLE = 24
        self.YELLOW = 25
        self.WHITE = 26
        self.RED = 31

        logging.addLevelName(self.RED, "*")
        logging.addLevelName(self.BLUE, "+")
        logging.addLevelName(self.GREEN, "-")
        logging.addLevelName(self.YELLOW, "!")
        logging.addLevelName(self.CYAN, ">")
        logging.addLevelName(self.PURPLE, "<")
        logging.addLevelName(self.WHITE, "/")

        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.INFO)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%Y_%m_%d")

        file_dir = os.getcwd() + '/log'

        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir
        self.log_name = self.log_path + "/" + log_cate + "." + self.log_time + '.log'
        # print(self.log_name)
        # print(self.log_name)

        # # 定义handler的输出格式
        # formatter = logging.Formatter(
        #     '[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

        formatter = ColoredFormatter(
            "%(log_color)s[%(asctime)s] [%(filename)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                '*': 'red',
                '+': 'blue',
                '-': 'green',
                '!': 'yellow',
                '>': 'cyan',
                '<': 'purple',
                '/': 'white'
            },
            secondary_log_colors={},
            style='%'
        )
        # fh = logging.FileHandler(self.log_name, 'a')  # 追加模式  这个是python2的
        fh = logging.FileHandler(self.log_name, 'a', encoding='utf-8')  # 这个是python3的
        # fh.setLevel(logging.INFO)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        #  添加下面一句，在记录日志之后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        fh.close()
        ch.close()

    def getlog(self):
        return self.logger

if __name__ == '__main__' :
    logger = Log('test_log',__name__).getlog()
    import utils.config
    # logger.setLevel(logging.INFO)

    logger.info('----------------------------------------------------------------------------------')
    logger.info('Epoch = {},  loss = {}'.format(1, 1))
    logger.log(21, 'on train set correct' )
    logger.log(22, 'on train set correct' )
    logger.log(23, 'on train set correct' )
    logger.log(24, 'on train set correct' )
    logger.log(25, 'on train set correct' )
    logger.log(26, 'on train set correct' )

    logger.log(31, 'on train set correct' )