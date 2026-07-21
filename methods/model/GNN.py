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


class myGAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha,
                 decode='distmult'):
        super(myGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, alpha=alpha))

        for l in range(1, num_layers):

            self.gat_layers.append(myGATConv(edge_dim, num_etypes,
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation, alpha=alpha))

        self.gat_layers.append(myGATConv(edge_dim, num_etypes,
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))
        self.register_buffer(
            "epsilon",
            torch.tensor([1e-12], dtype=torch.float32),
            persistent=False,
        )
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_classes * (num_layers + 2))
        elif decode == 'dot':
            self.decoder = Dot()
        elif decode == 'bilinear':
            self.decoder = BilinearDecoder(num_classes * (num_layers + 2))

    def l2_norm(self, x):

        return x / (torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))

    def encode(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](self.g, h, e_feat, res_attn=res_attn)
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)

        logits, _ = self.gat_layers[-1](self.g, h, e_feat, res_attn=res_attn)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        emb.append(logits)
        return torch.cat(emb, 1)

    def decode(self, node_repr, left, right, mid):
        left_emb = node_repr[left]
        right_emb = node_repr[right]
        return self.decoder(left_emb, right_emb, mid)

    def forward(self, features_list, e_feat, left, right, mid):
        node_repr = self.encode(features_list, e_feat)
        return self.decode(node_repr, left, right, mid)

    def forward_with_representation(self, features_list, e_feat, left, right, mid):
        """Return pair logits and the exact node representations used to decode them."""
        node_repr = self.encode(features_list, e_feat)
        return self.decode(node_repr, left, right, mid), node_repr

    def get_node_representation(self, features_list, e_feat):
        """Return the fused node representations used as decoder input."""
        return self.encode(features_list, e_feat)
