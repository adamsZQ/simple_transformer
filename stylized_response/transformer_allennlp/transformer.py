#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/11/19 7:38 pm
# @Author  : zchai
from typing import Dict

import torch
import torch.nn.functional as F
import math
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding
from allennlp.nn import util
from torch import nn

from stylized_response.utils import conf


class SimpleTransformer(Model):

    def __init__(self, vocab: Vocabulary):
        super().__init__(vocab)

        self.config = conf['transformer_allen']

        vocab_size = vocab.get_vocab_size('tokens')

        num_layers = self.config['num_layers']
        model_dim, self.max_length = self.config['model_dim'], self.config['max_length']
        head_num, dropout = self.config['head_num'], self.config['dropout']

        self.embed = EmbeddingLayer(
            vocab_size, model_dim, self.max_length,
        )
        # self.sos_token = nn.Parameter(torch.randn(model_dim))
        self.encoder = Encoder(num_layers, model_dim, vocab_size, head_num, dropout)
        self.decoder = Decoder(num_layers, model_dim, vocab_size, head_num, dropout)

        self.sos_token = nn.Parameter(torch.randn(model_dim))

        # TODO 为valid添加计算BLEU
        # self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        # self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        #
        # pad_index = self.vocab.get_token_index(self.vocab._padding_token, self._target_namespace)  # pylint: disable=protected-access
        # self._bleu = BLEU(exclude_indices={pad_index, self._end_index, self._start_index})

        self.loss_fn = nn.NLLLoss(reduction='none')

        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

    def forward(self, source_tokens, target_tokens, temperature=1.0):
        source_tokens_dict = source_tokens
        # target_tokens_dict = target_tokens
        source_tokens = source_tokens['tokens']
        target_tokens = target_tokens['tokens']
        # 移除第一个allennlp默认添加的<SOS>，因为transformer的decoder输入有<SOS>但是输出没有，所以下面输入添加了<SOS>
        target_tokens = target_tokens[:,1:]

        batch_size = source_tokens.size(0)
        max_enc_len = source_tokens.size(1)

        assert max_enc_len <= self.max_length

        lengths = util.get_text_field_mask(source_tokens_dict)

        pos_idx = torch.arange(self.max_length).unsqueeze(0).expand((batch_size, -1)).cuda(self.device)

        inp_lengths = (lengths == 1).long().sum(-1)

        pos_idx = pos_idx.to(inp_lengths.device)

        src_mask = pos_idx[:, :max_enc_len] >= inp_lengths.unsqueeze(-1)
        # src_mask = torch.cat((torch.zeros_like(src_mask[:, :1]), src_mask), 1)
        # src_mask = src_mask.view(batch_size, 1, 1, max_enc_len + 1)
        # FIXME 原版中这里为上面注释的那种，报错
        src_mask = src_mask.view(batch_size, 1, 1, max_enc_len)

        target_mask = torch.ones((self.max_length, self.max_length)).to(src_mask.device)
        target_mask = (target_mask.tril() == 0).view(1, 1, self.max_length, self.max_length)

        enc_input = self.embed(source_tokens, pos_idx[:, :max_enc_len])

        memory = self.encoder(enc_input, src_mask)

        sos_token = self.sos_token.view(1, 1, -1).expand(batch_size, -1, -1)

        # 和上面的mask是一个东西，但是用来计算loss用的，用allennlp的方便一点
        token_mask = util.get_text_field_mask(source_tokens_dict)

        if self.training:
            dec_input = target_tokens[:, :-1]
            max_dec_len = target_tokens.size(1)
            dec_input_emb = torch.cat((sos_token, self.embed(dec_input, pos_idx[:, :max_dec_len - 1])), 1)
            log_probs = self.decoder(
                dec_input_emb, memory,
                src_mask, target_mask[:, :, :max_dec_len, :max_dec_len],
                temperature
            )
            # FIXME style_transformer这里乘上了input_mask，但是对这应该是不需要
            loss = self.loss_fn(log_probs.transpose(1, 2), target_tokens)
            loss = loss.sum() / batch_size
            output_dict = {'loss': loss}
        else:

            log_probs = []
            next_token = sos_token
            prev_states = None

            for k in range(self.max_length):
                log_prob, prev_states = self.decoder.incremental_forward(
                    next_token, memory,
                    src_mask, target_mask[:, :, k:k + 1, :k + 1],
                    temperature,
                    prev_states
                )

                log_probs.append(log_prob)

                next_token = self.embed(log_prob.argmax(-1), pos_idx[:, k:k + 1])

                # if (pred_tokens == self.eos_idx).max(-1)[0].min(-1)[0].item() == 1:
                #    break

            # (batch_size,  max_sen_len(not max enc len), output_size)
            # target_tokens = (batch_size, max enc len)
            log_probs = torch.cat(log_probs, 1)
            # FIXME style_transformer这里乘上了input_mask，但是对这应该是不需要
            loss = self.loss_fn(log_probs.transpose(1, 2), target_tokens)
            loss = loss.sum() / batch_size
            output_dict = {'loss': loss, 'logits': log_probs}

        return output_dict


class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, vocab_size, head_num, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(model_dim, head_num, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(model_dim)

    def forward(self, x, mask):
        y = x

        assert y.size(1) == mask.size(-1)

        for layer in self.layers:
            y = layer(y, mask)

        return self.norm(y)


class Decoder(nn.Module):
    def __init__(self, num_layers, model_dim, vocab_size, head_num, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(model_dim, head_num, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(model_dim)
        self.generator = Generator(model_dim, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask, temperature):
        y = x

        assert y.size(1) == tgt_mask.size(-1)

        for layer in self.layers:
            y = layer(y, memory, src_mask, tgt_mask)

        return self.generator(self.norm(y), temperature)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, temperature, prev_states=None):
        y = x

        new_states = []

        for i, layer in enumerate(self.layers):
            y, new_sub_states = layer.incremental_forward(
                y, memory, src_mask, tgt_mask,
                prev_states[i] if prev_states else None
            )

            new_states.append(new_sub_states)

        new_states.append(torch.cat((prev_states[-1], y), 1) if prev_states else y)
        y = self.norm(new_states[-1])[:, -1:]

        return self.generator(y, temperature), new_states


class Generator(nn.Module):
    def __init__(self, model_dim, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(model_dim, vocab_size)

    def forward(self, x, temperature):
        return F.log_softmax(self.proj(x) / temperature, dim=-1)


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, model_dim, max_length):
        super(EmbeddingLayer, self).__init__()
        self.token_embed = Embedding(vocab_size, model_dim)
        self.pos_embed = Embedding(max_length, model_dim)
        self.vocab_size = vocab_size
        # if load_pretrained_embed:
        #     self.token_embed = nn.Embedding.from_pretrained(vocab.vectors)
        #     print('embed loaded.')

    def forward(self, x, pos):
        if len(x.size()) == 2:
            y = self.token_embed(x) + self.pos_embed(pos)
        else:
            y = torch.matmul(x, self.token_embed.weight) + self.pos_embed(pos)

        return y


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, head_num, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, head_num, dropout)
        self.pw_ffn = PositionwiseFeedForward(model_dim, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(model_dim, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.pw_ffn)


class DecoderLayer(nn.Module):
    def __init__(self, model_dim, head_num, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, head_num, dropout)
        self.src_attn = MultiHeadAttention(model_dim, head_num, dropout)
        self.pw_ffn = PositionwiseFeedForward(model_dim, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(model_dim, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.pw_ffn)

    def incremental_forward(self, x, memory, src_mask, tgt_mask, prev_states=None):
        new_states = []
        m = memory

        x = torch.cat((prev_states[0], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[0].incremental_forward(x, lambda x: self.self_attn(x[:, -1:], x, x, tgt_mask))
        x = torch.cat((prev_states[1], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[1].incremental_forward(x, lambda x: self.src_attn(x[:, -1:], m, m, src_mask))
        x = torch.cat((prev_states[2], x), 1) if prev_states else x
        new_states.append(x)
        x = self.sublayer[2].incremental_forward(x, lambda x: self.pw_ffn(x[:, -1:]))
        return x, new_states


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_num, dropout):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % head_num == 0
        self.d_k = model_dim // head_num
        self.head_num = head_num
        self.head_projs = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(3)])
        self.fc = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
                             for x, l in zip((query, key, value), self.head_projs)]

        attn_feature, _ = scaled_attention(query, key, value, mask)

        attn_concated = attn_feature.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)

        return self.fc(attn_concated)


def scaled_attention(query, key, value, mask):
    d_k = query.size(-1)
    scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(d_k)
    scores.masked_fill_(mask, float('-inf'))
    attn_weight = F.softmax(scores, -1)
    attn_feature = attn_weight.matmul(value)

    return attn_feature, attn_weight


class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.mlp = nn.Sequential(
            Linear(model_dim, 4 * model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            Linear(4 * model_dim, model_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class SublayerConnection(nn.Module):
    def __init__(self, model_dim, dropout):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x + self.dropout(y)

    def incremental_forward(self, x, sublayer):
        y = sublayer(self.layer_norm(x))
        return x[:, -1:] + self.dropout(y)


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim, eps=1e-6):
    m = nn.LayerNorm(embedding_dim, eps)
    return m



