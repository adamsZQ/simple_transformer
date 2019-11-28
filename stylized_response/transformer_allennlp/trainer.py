#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20/11/19 3:36 pm
# @Author  : zchai
import os

import torch
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.training import Trainer
from torch import optim

from stylized_response.my_logger import Logger
from stylized_response.transformer_allennlp.transformer import SimpleTransformer
from stylized_response.utils import conf


logger = Logger(__name__).get_logger()


class TransformerTrainer:

    def __init__(self, training=False):
        self.training = training
        config = conf['transformer_allen']
        prefix = config['data_prefix']
        train_file = config['train_data']
        valid_file = config['valid_data']
        self.model_path = config['model']

        if torch.cuda.is_available():
            cuda_device = 0
        else:
            cuda_device = -1

        self.reader = Seq2SeqDatasetReader(
                        source_tokenizer=WordTokenizer(),
                        target_tokenizer=WordTokenizer(),
                        source_token_indexers={'tokens': SingleIdTokenIndexer()},
                        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

        self.train_dataset = self.reader.read(os.path.join(prefix, train_file))
        self.valid_dataset = self.reader.read(os.path.join(prefix, valid_file))

        self.vocab = Vocabulary.from_instances(self.train_dataset + self.valid_dataset,
                                               min_count={'tokens': 3, 'target_tokens': 3})

        self.model = SimpleTransformer(self.vocab)

        optimizer = optim.Adam(self.model.parameters())

        iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
        # 迭代器需要接受vocab，在训练时可以用vocab来index数据
        iterator.index_with(self.vocab)

        self.model.cuda(cuda_device)

        patience = config['patience']
        epoch = config['epoch']
        self.trainer = Trainer(model=self.model,
                               optimizer=optimizer,
                               iterator=iterator,
                               patience=patience,
                               train_dataset=self.train_dataset,
                               validation_dataset=self.valid_dataset,
                               num_epochs=epoch,
                               cuda_device=cuda_device)

    def train(self):
        if self.training:
            self.vocab.save_to_files(os.path.join(self.model_path, 'vocab'))
            self.trainer.train()
        else:
            logger.warning('Model is not in training mode!')


    # def evaluate(self):
    #     if not self.training:
    #         final_metrics = evaluate(self.model, self.test_dataset, self.iterator, self.cuda_device, batch_weight_key=None)
    #         return final_metrics
    #     else:
    #         logger.warning('Mode is in training mode!')
