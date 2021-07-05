#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Kun Luo
# @Email   : olooook@outlook.com
# @File    : gcn.py
# @Date    : 2021/06/24
# @Time    : 18:00:15


import torch
from torch import nn


class GCNConv(nn.Module):
    """
    Implementation of first-order graph convolutional network
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(GCNConv, self).__init__()
        self.weights = nn.Parameter(torch.empty(out_channels, in_channels), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape is [batch, nodes, in_channels]
            adj (torch.Tensor): shape in [batch, nodes, nodes]
        Returns:
            [torch.Tensor]: shape is [batch, nodes, out_channels]
        """
        output = adj @ x @ self.weights.T
        if self.bias is not None:
            output += self.bias  # add bias
        return output
