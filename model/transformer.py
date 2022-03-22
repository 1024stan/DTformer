#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.masking import TriangularCausalMask, ProbMask
from model.encoder import Encoder, EncoderLayer, ConvLayer
from model.decoder import Decoder, DecoderLayer
from model.attention import FullAttention, ProbAttention, AttentionLayer
from model.embedding import DataEmbedding


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        # embedding
        # Encoding
        # self.enc_embedding = DataEmbedding(args.encoder_in_size, args.d_model, args.embed, args.data, args.dropout)
        # self.dec_embedding = DataEmbedding(args.decoder_in_size, args.d_model, args.embed, args.data, args.dropout)


        self.encoder = Encoder(encoder_in_size=args.encoder_in_size, d_model=args.d_model, embed_type=args.embed_type,
                               dropout=args.dropout, encoder_layers_num=args.encoder_layers_num, atten_type=args.atten_type, d_k=args.d_k,
                               d_v=args.d_v, n_heads=args.n_heads, d_ff=args.d_feedforward).cuda()

        self.decoder = Decoder(decoder_in_size=args.encoder_in_size, decoder_out_size=args.decoder_out_size, d_model=args.d_model,
                               embed_type=args.embed_type, data_len=args.data_len,atten_type=args.atten_type,
                               dropout=args.dropout, decoder_layer_num=args.decoder_layers_num,
                               d_k=args.d_k, d_v=args.d_v, n_heads=args.n_heads, d_ff=args.d_feedforward).cuda()
        self.projection = nn.Linear(args.d_model, args.decoder_out_size, bias=True).cuda()
        self.outlayer1 = nn.Linear(in_features=args.d_model, out_features=args.decoder_out_size, bias=True)
        self.outlayer2 = nn.Linear(in_features=args.decoder_out_size, out_features=args.decoder_out_size, bias=True)

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.outlayer1(dec_outputs)
        dec_logits = self.outlayer2(dec_logits)
        # dec_logits = dec_logits.view(-1, dec_logits.size(-1))

        # dec_logits = self.outlayer2(self.outlayer1(dec_outputs).permute(0, 2, 1))
        # dec_logits = dec_logits.float().cuda()

        # dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_outputs, enc_self_attns, dec_self_attns, dec_enc_attns