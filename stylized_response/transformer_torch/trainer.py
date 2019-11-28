#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/11/19 9:51 am
# @Author  : zchai
import copy
import time

import spacy
import torch
from torch import nn
from torchtext import data, datasets

from stylized_response.my_logger import Logger
from stylized_response.transformer_allennlp.transformer import Generator
from stylized_response.transformer_torch.evaluator import Evaluator
from stylized_response.transformer_torch.multi_gpu_loss import MultiGPULossCompute
from stylized_response.transformer_torch.tools import batch_size_fn
from stylized_response.transformer_torch.transformer import LabelSmoothing, NoamOpt, \
    greedy_decode, Batch, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, Encoder, \
    EncoderLayer, Decoder, DecoderLayer, Embeddings
from stylized_response.utils import conf

logger = Logger(__name__).get_logger()


class MyTrainer:

    def __init__(self, gpu_devices):
        self.config = conf['transformer_torch']
        self.devices = gpu_devices

        self.BOS_WORD = '<sos>'
        self.EOS_WORD = '<eos>'
        self.PAD_WORD = "<pad>"
        # due to unknown tag added to data.Field will lead to a unpredictable exception,
        # so set UNK_WORD == data.Field default unk tag
        self.UNK_WORD = "<unk>"

        self.evaluator = Evaluator()

        # model config
        self.num_layer = self.config['num_layer']
        self.d_model = self.config['d_model']
        self.d_ff = self.config['d_ff']
        self.num_head = self.config['num_head']
        self.dropout = self.config['dropout']

        # optimizer config
        self.factor = self.config['factor']
        self.warm_up = self.config['warm_up']
        self.lr = float(self.config['lr'])
        self.betas = self.config['betas']
        self.eps = float(self.config['eps'])

    def _load_data(self):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = data.Field(tokenize=tokenize_de, pad_token=self.PAD_WORD)
        TGT = data.Field(tokenize=tokenize_en, init_token=self.BOS_WORD,
                         eos_token=self.EOS_WORD, pad_token=self.PAD_WORD)

        self.MAX_LEN = self.config['max_length']
        self.train, self.val, self.test = datasets.IWSLT.splits(
            exts=('.de', '.en'), fields=(SRC, TGT),
            filter_pred=lambda x: len(vars(x)['src']) <= self.MAX_LEN and len(vars(x)['trg']) <= self.MAX_LEN)
        MIN_FREQ = self.config['min_frequency']
        SRC.build_vocab(self.train.src, min_freq=MIN_FREQ)
        TGT.build_vocab(self.train.trg, min_freq=MIN_FREQ)

        self.source_data = SRC
        self.target_data = TGT

    def train(self):
        if not hasattr(self, 'source_data'):
            logger.info('trainer does not have dataset, loading data now')
            self._load_data()
        pad_idx = self.target_data.vocab.stoi[self.PAD_WORD]
        model = self._make_model(len(self.source_data.vocab), len(self.target_data.vocab), N=self.num_layer,
                                 d_model=self.d_model, d_ff=self.d_ff, h=self.num_head, dropout=self.dropout)
        model.cuda()
        criterion = LabelSmoothing(size=len(self.target_data.vocab), padding_idx=pad_idx,
                                   smoothing=self.config['smoothing'])
        criterion.cuda()
        BATCH_SIZE = self.config['batch_size']
        train_iter = MyIterator(self.train, batch_size=BATCH_SIZE, device=torch.device('cuda'),
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=True)
        valid_iter = MyIterator(self.val, batch_size=BATCH_SIZE, device=torch.device('cuda'),
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                batch_size_fn=batch_size_fn, train=False)
        model_par = nn.DataParallel(model, device_ids=self.devices)

        model_opt = NoamOpt(model.src_embed[0].d_model, self.factor, self.warm_up,
                            torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps))
        MAX_EPOCH = self.config['max_epoch']
        for epoch in range(MAX_EPOCH):
            logger.info('Epoch:{}/{} training starts'.format(epoch, MAX_EPOCH))
            model_par.train()
            self._run_epoch((rebatch(pad_idx, b) for b in train_iter),
                            model_par,
                            MultiGPULossCompute(model.generator, criterion,
                                                devices=self.devices, opt=model_opt))
            model_par.eval()

            logger.info('---------------validation starts----------------')
            loss = self._run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                                   model_par,
                                   MultiGPULossCompute(model.generator, criterion,
                                                       devices=self.devices, opt=None))
            logger.info('validation average loss is {:.4f}'.format(loss))

            output_sen = []
            target_sen = []
            valid_iter_rebatch = (rebatch(pad_idx, b) for b in valid_iter)
            for i, batch in enumerate(valid_iter_rebatch):
                decode_max_length = self.config['decode_max_length']
                out = greedy_decode(model, batch.src, batch.src_mask,
                                    max_len=decode_max_length, start_symbol=self.target_data.vocab.stoi[self.BOS_WORD])
                output_sen += self._tensor2text(out)
                target_sen += self._tensor2text(batch.trg)

            bleu = self.evaluator.ref_bleu(output_sen, target_sen)

            logger.info('bleu result is {:.2f}'.format(bleu))

            # print a sample of valid
            for i, batch in enumerate(valid_iter):
                src = batch.src.transpose(0, 1)[:3]
                src_mask = (src != self.source_data.vocab.stoi[self.PAD_WORD]).unsqueeze(-2)
                out = greedy_decode(model, src, src_mask,
                                    max_len=self.MAX_LEN, start_symbol=self.target_data.vocab.stoi[self.BOS_WORD])
                print("Translation:", end="\t")
                for i in range(1, out.size(1)):
                    sym = self.target_data.vocab.itos[out[0, i]]
                    if sym == self.EOS_WORD:
                        break
                    print(sym, end=" ")
                print()
                print("Target:", end="\t")
                for i in range(1, batch.trg.size(0)):
                    sym = self.target_data.vocab.itos[batch.trg.data[i, 0]]
                    if sym == self.EOS_WORD:
                        break
                    print(sym, end=" ")
                print()
                break

    @staticmethod
    def _make_model(src_vocab, tgt_vocab, N=6,
                    d_model=512, d_ff=2048, h=8, dropout=0.1):
        """Helper: Construct a model from hyperparameters."""
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model

    def _run_epoch(self, data_iter, model, loss_compute):
        """Standard Training and Logging Function"""
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            out = model.forward(batch.src, batch.trg,
                                batch.src_mask, batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % self.config['print_steps'] == 1:
                elapsed = time.time() - start
                logger.info("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                            (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def _tensor2text(self, tensor):
        vocab = self.target_data.vocab
        tensor = tensor.cpu().numpy()
        text = []
        index2word = vocab.itos
        eos_idx = vocab.stoi[self.EOS_WORD]
        unk_idx = vocab.stoi[self.UNK_WORD]
        stop_idxs = [vocab.stoi['!'], vocab.stoi['.'], vocab.stoi['?']]
        for sample in tensor:
            sample_filtered = []
            prev_token = None
            for idx in list(sample):
                if prev_token in stop_idxs:
                    break
                if idx == unk_idx or idx == prev_token or idx == eos_idx:
                    continue
                prev_token = idx
                sample_filtered.append(index2word[idx])

            sample = ' '.join(sample_filtered)
            text.append(sample)

        return text


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    """Fix order in torchtext to match ours"""
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)
