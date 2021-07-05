#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : gat.py
# @Date    : 2021/06/25
# @Time    : 18:14:39


import torch
from torch import nn


class GraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=.1, alpha=.2, bias=True):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.empty([out_channels, in_channels]), requires_grad=True)
        self.a = nn.Parameter(torch.empty(2, out_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True) if bias else None
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape in [batch, nodes, in_channels]
            mask (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.

        Returns:
            [torch.Tensor]: shape is [batch, nodes, out_channels]
        """
        hidden = x @ self.W.T  # [batch, nodes, out_channels]
        attn1, attn2 = torch.unbind(hidden @ self.a.T, -1)  # [batch, nodes]
        # [batch, nodes, 1] + [batch, 1, nodes] => [batch, nodes, nodes]
        attn = attn1.unsqueeze(-1) + attn2.unsqueeze(1)
        # leaky ReLU
        attn = self.leaky_relu(attn)
        # mask
        if mask is not None:
            attn += torch.where(mask > 0, 0, -1e12)
        # softmax
        attn = torch.softmax(attn, dim=-1)  # [batch, nodes, nodes]
        # dropout
        attn, hidden = self.dropout(attn), self.dropout(hidden)
        output = attn @ hidden  # [batch, nodes, out_channels]
        # add bias
        if self.bias is not None:
            output += self.bias
        return output  # [batch, nodes, out_channels]


class GAT(nn.Module):
    """
    Graph Attention Network
    """

    def __init__(self, in_channels, out_channels,
                 n_heads=8, dropout=.1, alpha=.2, bias=True, aggregate='concat'):
        """
        Args:
            in_channels ([type]): input channels
            out_channels ([type]): output channels
            n_heads (int, optional): number of heads. Defaults to 64.
            dropout (float, optional): dropout rate. Defaults to .1.
            alpha (float, optional): leaky ReLU negative_slope. Defaults to .2.
            bias (bool, optional): use bias. Defaults to True.
            aggregate (str, optional): aggregation method. Defaults to 'concat'.
        """
        super(GAT, self).__init__()
        assert aggregate in ['concat', 'average']
        self.attns = nn.ModuleList([
            GraphAttention(in_channels, out_channels, dropout=dropout, alpha=alpha, bias=bias)
            for _ in range(n_heads)
        ])
        self.aggregate = aggregate

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): shape is [batch, nodes, in_channels]
            adj (torch.Tensor, optional): shape is [batch, nodes, nodes]. Defaults to None.
        """
        if self.aggregate == 'concat':
            output = torch.cat([attn(x, adj) for attn in self.attns], dim=-1)
        else:
            output = sum([attn(x, adj) for attn in self.attns]) / len(self.attns)
        return torch.relu(output)
