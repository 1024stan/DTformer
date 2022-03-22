#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import numpy as np


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q, data_embed_size = seq_q.size()
    batch_size, len_k, data_embed_size = seq_k.size()
    # eq(zero) is PAD token
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    # pad_attn_mask = seq_k.data.eq(0)
    # return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]
    pad_atten_mask = (torch.triu(torch.ones(batch_size, len_q, len_k)) == 1).transpose(0,1)
    pad_atten_mask = pad_atten_mask.float().masked_fill(pad_atten_mask == 0, float('-inf')).masked_fill(pad_atten_mask == 1, float(0.0))
    return pad_atten_mask.permute(1, 0, 2).bool().cuda()

def get_attn_subsequence_mask(seq):
    '''
    seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dytpe=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask