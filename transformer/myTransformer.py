import copy
import sys
from typing import Optional, Union

import torch
from torch import nn, device
import torch.nn.functional as F
import numpy as np
import math



class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionEmbedding, self).__init__()
        self.encodedMatrix = torch.zeros(max_len, d_model, device="cuda:1")
        pos = torch.arange(0, max_len, device="cuda:1")
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device="cuda:1").float()
        self.encodedMatrix[:, 0::2] = torch.sin(pos/10000**(_2i/d_model))
        self.encodedMatrix[:, 1::2] = torch.cos(pos/10000**(_2i/d_model))
        # for i in range(max_len):
        #     for j in range(d_model):
        #         if j % 2 ==0:
        #             self.encodedMatrix[i, j] = math.sin(i/pow(10000, j/d_model))
        #         else:
        #             self.encodedMatrix[i, j] = math.cos(i/pow(10000, (j-1)/d_model))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encodedMatrix[:seq_len, :]

# input data (seq_len,)
# after tokenEmbedding (seq_len, d_model)
# after posEmbedding (seq_len, d_model)   posEncodeing (max_len, d_model)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, pad_idx):
        super(TransformerEmbedding, self).__init__()
        self.token_embed = TokenEmbedding(vocab_size, d_model, pad_idx)
        self.pos_embed = PositionEmbedding(max_len, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        token_embedMatrix = self.token_embed(x)
        pos_embedMatrix = self.pos_embed(x)
        pos_embedMatrix = pos_embedMatrix.to(token_embedMatrix.device)
        res = self.drop_out(token_embedMatrix + pos_embedMatrix)
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, drop_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // head_num
        self.d_v = self.d_k
        self.head_num = head_num
        self.Q_Linear = nn.Linear(d_model, d_model, bias=False)
        self.K_Linear = nn.Linear(d_model, d_model, bias=False)
        self.V_Linear = nn.Linear(d_model, d_model, bias=False)
        self.final_Linear = nn.Linear(d_model,d_model)
        self.attn = None
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, type, mask=None, attention_mask=None):
        # data (batch_size, seq_len, d_model)
        if mask is not None:
            # raw mask (batch_size, seq_len, seq_len)  need alignment
            mask = mask.unsqueeze(1).expand(-1, self.head_num, -1, -1)
            # processed mask (batch_size, head_num, seq_len, seq_len)

        if attention_mask is not None:
            # raw mask (batch_size, seq_len) 
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.head_num, -1).unsqueeze(2).expand(-1, -1, query.size(1), -1)
            # processed mask (batch_size, head_num, seq_len, seq_len)


        batch_size = query.size(0)
        Query = self.Q_Linear(query).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        Key = self.K_Linear(key).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        Value = self.V_Linear(value).view(batch_size, -1, self.head_num, self.d_v).transpose(1, 2)
        # data (batch_size, seq_len, head_num, d_k)->(batch_size, head_num, seq_len, d_k)

        # Attention
        score = torch.matmul(Query, Key.transpose(2, 3))/math.sqrt(self.d_k)
        # (batch_size, head_num, seq_len, seq_len)
        if mask is not None and attention_mask is not None and type == "decoder":
            mask = mask & attention_mask
            score = score.masked_fill(mask == 0, -1e9)

        if attention_mask is not None and type == "encoder":
            score = score.masked_fill(attention_mask == 0, -1e9)

        attn_weight = F.softmax(score, dim=-1)
        attn_weight = self.drop_out(attn_weight)
        res = torch.matmul(attn_weight, Value)
        # (batch_size, seq_len, head_num, d_k)
        res = res.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)
        return self.drop_out(self.final_Linear(res))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob):
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        res = self.l1(x)
        res = F.relu(res)
        res = self.drop_out(res)
        res = self.l2(res)
        return self.drop_out(res)


class AddAndLayerNorm(nn.Module):
    def __init__(self, features, drop_prob):
        super(AddAndLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(features)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sublayer):
        res = self.dropout(sublayer) + x
        res = self.layernorm(res)
        
        return res


class EncoderBlock(nn.Module):
    def __init__(self, d_model, drop_prob, head_nums, d_ff):
        super(EncoderBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, head_nums, drop_prob)
        self.add_and_norm = AddAndLayerNorm(d_model, drop_prob)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, drop_prob)


    def forward(self, x, mask):
        res = self.multi_head_attn(x, x, x, "encoder", attention_mask=mask)
        res1 = self.add_and_norm(x, res)
        res = self.feed_forward(res1)
        return self.add_and_norm(res1, res)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, head_nums, drop_prob, block_nums,
                 max_len, pad_idx):
        super(Encoder, self).__init__()
        self.blocks_nums = block_nums
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, pad_idx)
        self.blks = nn.ModuleList([EncoderBlock(d_model, drop_prob, head_nums, d_ff) for _ in range(block_nums)])


    def forward(self, x, attention_mask):
        res = self.embed(x)
        for block in self.blks:
            res = block(res, attention_mask)
    
        return res


class DecoderBlock(nn.Module):
    def __init__(self, d_model, drop_prob, head_nums, d_ff):
        super(DecoderBlock, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, head_nums, drop_prob)
        self.add_and_norm = AddAndLayerNorm(d_model, drop_prob)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, drop_prob)

    def forward(self, x, encoder_res, mask=None):
        X = x
        sublayer = self.multi_head_attn(X, X, X, "decoder", mask)
        X = self.add_and_norm(X, sublayer)
        sublayer = F.relu(self.multi_head_attn(X, encoder_res, encoder_res, "decoder"))
        X = self.add_and_norm(X, sublayer)
        sublayer = self.feed_forward(X)
        return self.add_and_norm(X, sublayer)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, block_nums, head_nums, d_ff, pad_idx):
        super(Decoder, self).__init__()
        self.embed = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, pad_idx)
        self.DecoderBlocks = nn.ModuleList([DecoderBlock(d_model, drop_prob, head_nums, d_ff) for _ in range(block_nums)])


    def forward(self, X, encoder_res, mask=None):
        res = self.embed(X)
        for block in self.DecoderBlocks:
            res = block(res, encoder_res, mask)
        return res



class myTransformerClass(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob,
                 block_nums, head_nums, d_ff, d_output, pad_idx
                 ):
        super(myTransformerClass, self).__init__()
        self.Encoder = Encoder(vocab_size, d_model, d_ff, head_nums,
                               drop_prob, block_nums, max_len, pad_idx)
        self.Decoder = Decoder(vocab_size, d_model, max_len, drop_prob,
                               block_nums, head_nums, d_ff, pad_idx)
        self.LinearOut = nn.Linear(d_model, d_output)


    def forward(self, source, target, mask=None, attention_mask=None):
        source_res = self.Encoder(source, attention_mask)
        target_res = self.Decoder(target, source_res, mask)
        # output is (batch_size, seq_len, d_model)
        final_res = self.LinearOut(target_res)
        # output is (batch_size, seq_len, d_output)
        return final_res







