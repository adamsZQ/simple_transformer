#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20/11/19 3:34 pm
# @Author  : zchai
import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../')))

from stylized_response.transformer_allennlp.trainer import TransformerTrainer


def train():
    model = TransformerTrainer(training=True)
    model.train()


if __name__ == '__main__':
    train()