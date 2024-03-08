import copy
import torch
from torch import nn
from .base_function import *

class TTM(nn.Module):
    """
    Texture refinement network (TTM)
    :param d_model: number of channels in input
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model=512, dim_feedforward=2048,
                 activation="LeakyReLU",
                 affine=True, norm='instance'):
        super().__init__()
        encoder_layer = SPVT(d_model, dim_feedforward, activation, affine, norm)
        if norm == 'batch':
            encoder_norm = None
            decoder_norm = nn.BatchNorm1d(d_model, affine=affine)
        elif norm == 'instance':
            encoder_norm = None
            decoder_norm = nn.InstanceNorm1d(d_model, affine=affine)

        self.encoder = SPVTS(encoder_layer, encoder_norm)

        decoder_layer = TPVT(d_model, dim_feedforward, activation, affine, norm)

        self.decoder = TPVTs(decoder_layer, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = 2

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, val, pos_embed=None, test=False):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        tgt = tgt.flatten(2).permute(2, 0, 1)
        val = val.flatten(2).permute(2, 0, 1)
        if pos_embed != None:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, val, pos=pos_embed)

        return hs.view(bs, c, h, w)


class SPVTS(nn.Module):
    """
    Source pose variation texture (SPVTs)
    SPVTS = 2 * SPVT
    """
    def __init__(self, encoder_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer)
        self.norm = norm

    def forward(self, src, pos = None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0)).permute(2, 0, 1)

        return output


class TPVTs(nn.Module):
    """
    Target pose variation texture (TPVTs)
    TPVTS =  2 * TPVT
    """
    def __init__(self, decoder_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer)
        self.norm = norm

    def forward(self, tgt, memory, val, pos = None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, val, pos=pos)
        if self.norm is not None:
            output = self.norm(output.permute(1, 2, 0))
        return output


class SPVT(nn.Module):
    """
    Context Augment Block (CAB)
    :param d_model: number of channels in input
    :param nhead: number of heads in attention module
    :param dim_feedforward: dimension in feedforward
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    :param affine: affine in normalization
    :param norm: normalization function 'instance, batch'
    """
    def __init__(self, d_model, dim_feedforward=2048,
                 activation="LeakyReLU", affine=True, norm='instance'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm2 = nn.BatchNorm1d(d_model, affine=affine)
        else:
            self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = get_nonlinearity_layer(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + src2
        src = self.norm1(src.permute(1, 2, 0)).permute(2, 0, 1)
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src.permute(1, 2, 0)).permute(2, 0, 1)
        return src


class TPVT(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048,
                 activation="LeakyReLU", affine=True, norm='instance'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, 2)
        self.multihead_attn = nn.MultiheadAttention(d_model, 2)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if norm == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm2 = nn.BatchNorm1d(d_model, affine=affine)
            self.norm3 = nn.BatchNorm1d(d_model, affine=affine)
        else:
            self.norm1 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm2 = nn.InstanceNorm1d(d_model, affine=affine)
            self.norm3 = nn.InstanceNorm1d(d_model, affine=affine)

        self.activation = get_nonlinearity_layer(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, val, pos = None):
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, pos), key=self.with_pos_embed(memory, pos), value=val)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        return tgt

def _get_clones(module):
    return nn.ModuleList([copy.deepcopy(module) for i in range(2)])


