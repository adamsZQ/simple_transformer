#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/11/19 10:14 am
# @Author  : zchai
import argparse
import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../../')))

from stylized_response.transformer_torch.trainer import MyTrainer
from stylized_response.my_logger import Logger


logger = Logger(__name__).get_logger()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_list', nargs='+', type=int)
    args = parser.parse_args()
    gpu_list = args.gpu_list
    logger.info('program will run on gpu {}'.format(gpu_list))
    trainer = MyTrainer(gpu_list)
    trainer.train()


if __name__ == '__main__':
    train()
