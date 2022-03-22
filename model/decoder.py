#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding import DataEmbedding
from model.attention import MultiHeadAttention
from model.encoder import PoswiseFeedForwardNet
from tools.masking import get_attn_pad_mask, get_attn_subsequence_mask
from model.attentions.SEAttention import SEAttention
from model.attentions.ECAAttention import ECAAttention
from model.attentions.ResidualAttention import ResidualAttention
from model.attentions.ExternalAttention import ExternalAttention
from model.attentions.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention



class DecoderLayer(nn.Module):
    def __init__(self, atten_type, d_model, d_k, d_v, n_heads, d_ff):
        super(DecoderLayer, self).__init__()

        if atten_type == 'multi':
            self.decoder_self_atten = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
            self.decoder_encoder_atten = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        elif atten_type == 'SENet':
            self.decoder_self_atten = SEAttention(channel=d_model, reduction=n_heads)
            self.decoder_encoder_atten = SEAttention(channel=d_model, reduction=n_heads)
        elif atten_type == 'ECA':
            self.decoder_self_atten = ExternalAttention(d_model=d_model,S=8)
            self.decoder_encoder_atten = ExternalAttention(d_model=d_model,S=8)
        elif atten_type == 'ResAtten':
            self.decoder_self_atten = SimplifiedScaledDotProductAttention(d_model=d_model, h=8)
            self.decoder_encoder_atten = SimplifiedScaledDotProductAttention(d_model=d_model, h=8)


        # self.decoder_self_atten = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        # self.decoder_encoder_atten = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)  # d_ff : feed_forward层中间层神经元个数
    def forward(self, decoder_inputs, encoder_outputs,
                decoder_self_atten_mask, decoder_encoder_atten_mask):
        '''
               dec_inputs: [batch_size, tgt_len, d_model]
               enc_outputs: [batch_size, src_len, d_model]
               dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
               dec_enc_attn_mask: [batch_size, tgt_len, src_len]
               '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        decoder_outputs, decoder_self_atten = self.decoder_self_atten(decoder_inputs, decoder_inputs, decoder_inputs, decoder_self_atten_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        decoder_outputs, decoder_encoder_atten = self.decoder_encoder_atten(decoder_outputs, encoder_outputs, encoder_outputs, decoder_encoder_atten_mask)
        decoder_outputs = self.pos_ffn(decoder_outputs)
        return decoder_outputs, decoder_self_atten, decoder_encoder_atten




class Decoder(nn.Module):
    def __init__(self, decoder_in_size, decoder_out_size, d_model, embed_type,
                 dropout, decoder_layer_num, d_k, d_v, n_heads, d_ff, data_len, atten_type='multi'):

        super(Decoder, self).__init__()
        self.embedding = DataEmbedding(c_in=decoder_in_size, d_model=d_model, embed_type=embed_type, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(atten_type=atten_type, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, d_ff=d_ff)
                                     for _ in range(decoder_layer_num)])


    def forward(self, decoder_inputs, encoder_inputs, encoder_outputs):
        decoder_outputs = self.embedding(decoder_inputs, decoder_inputs)
        decoder_self_atten_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_inputs)
        decoder_self_atten_subseq_mask = get_attn_subsequence_mask(decoder_inputs)
        decoder_self_atten_mask = torch.gt((decoder_self_atten_pad_mask.cuda() + decoder_self_atten_subseq_mask.cuda()), 0).cuda()
        decoder_encoder_atten_mask = get_attn_pad_mask(decoder_inputs, encoder_inputs)
        decoder_self_attens, decoder_encoder_attens = [], []
        for layer in self.layers:
            decoder_outputs, decoder_self_atten, decoder_encoder_atten = layer(decoder_outputs,encoder_outputs,
                                                                               decoder_self_atten_mask, decoder_encoder_atten_mask)
            decoder_self_attens.append(decoder_self_atten)
            decoder_encoder_attens.append(decoder_encoder_atten)
        return decoder_outputs, decoder_self_attens, decoder_encoder_attens





