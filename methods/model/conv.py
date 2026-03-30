"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class myGATConv(nn.Module):
    """
    Modified with Relation-aware Attention and type fix
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        # 关系类型嵌入和权重
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.w_r = nn.Embedding(num_etypes, 1)  # 关系感知注意力权重
        nn.init.constant_(self.w_r.weight, 1.0)  # 初始化为1

        # 特征变换层
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # 边特征处理
        self.fc_e = nn.Sequential(
            nn.Linear(edge_feats, edge_feats * 2),
            nn.ReLU(),
            nn.Linear(edge_feats * 2, edge_feats * num_heads)
        )

        # 注意力参数
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))

        # 正则化
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # 残差连接
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        # 初始化
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        """参数初始化"""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        for m in self.fc_e:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            # 检查0入度节点
            if not self._allow_zero_in_degree and (graph.in_degrees() == 0).any():
                raise DGLError('0-in-degree nodes detected. Use add_self_loop() or set allow_zero_in_degree=True.')

            # 特征变换
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            # 边特征处理（关键修改点）
            r_ids = e_feat.long()  # 强制转换为long类型
            e_emb = self.edge_emb(r_ids)  # 使用修正后的边类型ID
            e_feat_processed = self.fc_e(e_emb).view(-1, self._num_heads, self._edge_feats)

            # 计算注意力各部分
            ee = (e_feat_processed * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            # 消息传递
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))

            # 关系感知注意力权重应用（关键修改点）
            w = self.w_r(r_ids).view(-1, 1, 1)  # [E,1,1]
            e = e * w  # 应用关系权重

            # 计算softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha

            # 聚合消息
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']

            # 残差连接
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
                rst = rst + feat_dst  # 双向残差

            # 偏置和激活
            if self.bias:
                rst += self.bias_param
            if self.activation:
                rst = self.activation(rst)

            return rst, graph.edata.pop('a').detach()

class NeighborRoutingConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_factors, dropout=0.0, negative_slope=0.2):
        super(NeighborRoutingConv, self).__init__()
        self.K = num_factors
        self.d_k = out_dim // num_factors
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.Tensor(self.K, self.d_k))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.attn)
    def forward(self, g, h):
        Wh = self.linear(h)  # [N, out_dim]
        Wh = Wh.view(-1, self.K, self.d_k)  # [N, K, d_k]
        with g.local_scope():
            g.ndata['h'] = Wh
            out_list = []
            for k in range(self.K):
                g.ndata['h_k'] = Wh[:, k, :]
                g.ndata['a_l'] = (Wh[:, k, :] * self.attn[k]).sum(dim=1, keepdim=True)
                g.apply_edges(lambda edges: {
                    'e': self.leaky_relu(edges.src['a_l'] + edges.dst['a_l'])
                })
                g.edata['alpha'] = dgl.nn.functional.edge_softmax(g, g.edata['e'])
                g.update_all(
                    message_func=dgl.function.u_mul_e('h_k', 'alpha', 'm'),
                    reduce_func=dgl.function.sum('m', 'h_agg')
                )
                h_new = g.ndata['h_agg']
                h_new = self.dropout(h_new)
                out_list.append(h_new.unsqueeze(1))
            h_out = torch.cat(out_list, dim=1)
            return h_out  # [N, K, d_k]
