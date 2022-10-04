# -*- coding: utf-8 -*-
"""
   File Name：     GCNlayers
   Description : 图卷积层，多个channal
   date：          2021/3/11
"""
__author__ = 'mingjian'

import torch
import torch.nn as nn
import torch_geometric.nn as gnn

gnn_dict = {
    "GCN": gnn.GCNConv,
    "GraphSAGE": gnn.SAGEConv,
    "GAT": gnn.GATConv
}


class MultiGCNLayers(nn.Module):

    def __init__(self, d_in, d_h, d_out, sz_c, sz_l, drop_rate, device, g_name="GCN", g_norm=False, alpha=None):
        super().__init__()
        self.alpha = alpha
        self.sz_c = sz_c
        d_in = d_in
        d_h = d_h
        self.d_out = d_out
        self.sz_l = sz_l
        self.drop_rate = drop_rate
        self.device = device
        g_name = g_name
        self.gcn_layer = nn.ModuleList(self.channal_block(d_in, d_h, d_out, g_name, g_norm))
        self.layer_norm = nn.LayerNorm(d_out, eps=1e-6)
        # self.all_smu = nn.ModuleList([SMULayer() for _ in range(self.sz_c)])
        # self.Dropout = nn.Dropout(self.drop_rate)

    def reset_parameters(self):
        for layer in self.gcn_layer:
            for f in layer:
                if type(f) in list(gnn_dict.values()):
                    f.reset_parameters()
                if type(f) == gnn.BatchNorm:
                    f.reset_parameters()

    def channal_block(self, d_in, d_h, d_out, g_name, g_norm):
        layer = []
        for c in range(self.sz_c):
            t_layer = []
            for l in range(self.sz_l):
                if l == 0:
                    t_layer.append(self.get_gnn(d_in, d_h, g_name))
                elif l == (self.sz_l - 1):
                    t_layer.append(self.get_gnn(d_h, d_out, g_name))
                else:
                    t_layer.append(self.get_gnn(d_h, d_h, g_name))
                t_layer.append(nn.Dropout(self.drop_rate))
                if g_norm:
                    t_layer.append(gnn.GraphSizeNorm())
                    if l == (self.sz_l - 1):
                        t_layer.append(gnn.BatchNorm(d_out))
                    else:
                        t_layer.append(gnn.BatchNorm(d_h))
                t_layer.append(nn.ReLU())
            layer.append(nn.ModuleList(t_layer))
            # layer.append(nn.Sequential(*t_layer))
        return layer
        # pass

    def get_gnn(self, d_in, d_out, g_name):
        """"""
        if g_name == 'GAT':
            return gnn.GATConv(d_in, d_out // 8, 8)  # 8 head
        else:
            return gnn_dict[g_name](d_in, d_out)

    def forward(self, x, edge, batch, all_smu=None):
        # channal = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
        smu = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
        # channal = []
        if all_smu is not None:
            h_smu = x
        h = x
        for i, layer in enumerate(self.gcn_layer):
            for f in layer:
                rh = h
                if type(f) in list(gnn_dict.values()):
                    h = f(h, edge)
                else:
                    h = f(h)
                if isinstance(f, nn.ReLU):
                    if all_smu is not None:
                        h_smu = all_smu[i](h_smu, h)
                        if i == self.sz_l - 1 and self.alpha is not None:
                            h_smu = self.alpha * h + float(1.0-self.alpha) * h_smu

            # channal[i] = h
            if all_smu is not None:
                smu[i] = h_smu
            else:
                smu[i] = h
        # channal =
        batchs = torch.ones([self.sz_c, batch.size(0)]).to(self.device) * batch
        return self.layer_norm(smu), batchs



    # def forward(self, x, edge, batch):
    #     channal = torch.empty([self.sz_c, x.size(0), self.d_out]).to(self.device)
    #     # channal = []
    #     for i in range(self.sz_c):
    #         h = x
    #         m = self.gcn_layer[i]
    #         for el in m:
    #             rh = h
    #             if isinstance(el, self.gnn_type):
    #                 h = el(h, edge)
    #             else:
    #                 h = el(h)
    #             if isinstance(el, nn.ReLU) and h.size(1) == rh.size(1):
    #                 h = h + rh  # 残差
    #
    #         channal[i] = h
    #     # channal =
    #     batchs = torch.ones([self.sz_c, batch.size(0)]).to(self.device) * batch
    #     return self.layer_norm(channal), batchs

# print(isinstance(model,MultiGCNLayers))
if __name__ == '__main__':
    model = MultiGCNLayers(10,32,2,3,3,0.2,"cuda:0")
    print(model)