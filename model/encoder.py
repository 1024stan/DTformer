#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding import DataEmbedding
from model.attention import MultiHeadAttention
from tools.masking import get_attn_pad_mask
from model.attentions.SEAttention import SEAttention
from model.attentions.ExternalAttention import ExternalAttention
from model.attentions.ECAAttention import ECAAttention
from model.attentions.ResidualAttention import ResidualAttention
from model.attentions.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class informwer_EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(informwer_EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask = attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y)

class informwer_Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(informwer_Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
            x = self.attn_layers[-1](x)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, atten_type, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        if atten_type == 'multi':
            self.enc_self_atten = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        elif atten_type == 'SENet':
            self.enc_self_atten = SEAttention(channel=d_model, reduction=n_heads)
        elif atten_type == 'ECA':
            self.enc_self_atten = ExternalAttention(d_model=d_model,S=8)
        elif atten_type == 'ResAtten':
            self.enc_self_atten = SimplifiedScaledDotProductAttention(d_model=d_model, h=8)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff) # d_ff : feed_forward层中间层神经元个数

    def forward(self, enc_inputs, enc_self_atten_mask):
        '''
                enc_inputs: [batch_size, src_len, d_model]
                enc_self_attn_mask: [batch_size, src_len, src_len]
                '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_atten(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_atten_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, encoder_in_size, d_model, embed_type='fixed',
                 dropout=0.0, encoder_layers_num=2, d_k=128, d_v=128, n_heads=64, d_ff=256,atten_type='multi'):
        super(Encoder, self).__init__()
        self.embeding = DataEmbedding(c_in=encoder_in_size, d_model=d_model, embed_type=embed_type, dropout=dropout)
        self.layers = nn.ModuleList([EncoderLayer(atten_type=atten_type, d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads, d_ff=d_ff)
                                     for _ in range(encoder_layers_num)])

    def forward(self, encoder_inputs):
        encoder_outputs = self.embeding(encoder_inputs, encoder_inputs)
        enc_self_atten_mask = get_attn_pad_mask(encoder_inputs, encoder_inputs)
        enc_self_attens = []
        for layer in self.layers:
            encoder_outputs, enc_self_atten = layer(encoder_outputs, enc_self_atten_mask)
            enc_self_attens.append(enc_self_atten)
        return encoder_outputs, enc_self_attens



