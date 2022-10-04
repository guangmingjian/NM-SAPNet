#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/13 16:45
# @Version : 1.0
# @File    : SimilarMeasure.py
import torch


class SimilarMeasure():
    """"""

    def __init__(self, agg="sum"):
        super(SimilarMeasure, self).__init__()
        """"""
        self.agg = agg

    def correlation_loss(self, z: torch.tensor):
        """"""
        # sz_c,d = z.size(0),z.size(-1)
        z_ = z.mean(dim=-1, keepdim=True)
        z_stds = z.std(dim=-1)
        all_t = None
        us = None
        for m in (z - z_):
            if all_t is None:
                all_t = m
            else:
                all_t = all_t * m
        for u in z_stds:
            if us is None:
                us = u
            else:
                us = us * u
        sim = all_t.sum(dim=-1) / (z.size(-1) - 1) / us
        # sim_dis = 1 - torch.abs(sim)
        if self.agg == 'sum':
            return 2 * torch.sum(sim ** 2)
        else:
            return 2 * torch.mean(sim ** 2)
