import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from conv import myGATConv

class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()

class Dot(nn.Module):
    def __init__(self):
        super(Dot, self).__init__()
    def forward(self, left_emb, right_emb, r_id):
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(left_emb, right_emb).squeeze()


class BilinearDecoder(nn.Module):
    def __init__(self, input_dim):
        super(BilinearDecoder, self).__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1, bias=True)

    def forward(self, left_emb, right_emb, r_id=None):
        return self.bilinear(left_emb, right_emb).squeeze()


from conv import NeighborRoutingConv  # 确保你已粘贴该层到conv.py并import

class MyFactorGNN(nn.Module):
    def __init__(self, in_dim, num_hidden=64, num_factors=2, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(NeighborRoutingConv(
                in_dim if i == 0 else num_hidden,
                num_hidden,
                num_factors,       # 这里就是K=2
                dropout=0.2
            ))
        self.out_dim = num_hidden
        self.num_factors = num_factors

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)                # [N, 2, 32]（每层）
            x = x.reshape(x.shape[0], -1)  # [N, 64]
        return x  # 送入后续分类/正则损失




