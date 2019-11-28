#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27/11/19 9:38 am
# @Author  : zchai
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


class Evaluator(object):

    @staticmethod
    def _nltk_bleu(source, target):
        source = [word_tokenize(source.lower().strip()) for source in source]
        target = word_tokenize(target.lower().strip())
        return sentence_bleu(source, target) * 100

    def ref_bleu(self, source, target):
        assert len(source) == len(target), 'Batch Size of inputs does not match!'
        sum_ = 0
        n = len(source)
        for x, y in zip(source, target):
            sum_ += self._nltk_bleu([x], y)
        return sum_ / n
