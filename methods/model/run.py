# 在6的基础上修改，训练batch抽取的负样本不会包含正样本，加入困难采样
import json
import sys
sys.path.append('../../') #将上级目录的上两级路径添加到 Python 模块搜索路径中
import time
import argparse

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT
import dgl
import os
from functools import reduce
import scipy.sparse as sp


def sp_to_spt(mat): #转换矩阵为张量
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat): # 判断numpy数组和稀疏矩阵，转化成张量
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def factorize(h, num_factors):
    B, D = h.shape
    factor_dim = D // num_factors
    return h.view(B, num_factors, factor_dim)  # [B, K, D//K]

def factor_compactness(factor_tensor):
    # 输入: [B, K, D]，对每个 K 求均值，然后求与每个样本的距离
    B, K, D = factor_tensor.shape
    loss = 0
    for i in range(K):
        f = factor_tensor[:, i, :]  # [B, D]
        mean = f.mean(dim=0, keepdim=True)  # [1, D]
        loss += F.mse_loss(f, mean.expand_as(f))
    return loss / K

def factor_irrelevance(factor_tensor):
    # 保证不同因子解耦：相似性越低越好
    B, K, D = factor_tensor.shape
    if K < 2:
        # 当 K=1 时无需计算解耦损失，直接返回 0（保持 dtype/device 一致）
        return torch.zeros((), dtype=factor_tensor.dtype, device=factor_tensor.device)
    loss = 0
    for i in range(K):
        for j in range(i+1, K):
            fi = factor_tensor[:, i, :]
            fj = factor_tensor[:, j, :]
            cos = F.cosine_similarity(fi, fj, dim=-1)  # [B]
            loss += cos.abs().mean()
    return loss * 2 / (K * (K - 1))

def cosine_similarity_matrix(feat):
    # feat: [N, d]
    normed = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
    return normed @ normed.t()

def inter_node_similarity_loss(H, X):
    # H: [N, D]（解耦新空间特征），X: [N, d_in]（原始特征）
    S_H = cosine_similarity_matrix(H)
    S_X = cosine_similarity_matrix(X)
    return ((S_H - S_X) ** 2).mean()


def run_model_DBLP(args):
    # 记录训练的参数信息，在训练开始时
    log_file_path = './training_logs_banjiandu3.jsonl'
    start_rec = {
        "phase": "start",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "neg_ratio": args.neg_ratio,
        "pos_weight": args.pos_weight,
        "hardneg_enabled": args.hardneg,
		"factor_K": args.factor_K,
    	"lambda_c": args.lambda_c,
    	"lambda_i": args.lambda_i,
    }
    append_json_line(log_file_path, start_rec)

    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]

    print("len(features_list) =", len(features_list))
    print("node counts =", dl.nodes.get('count', 'NA'))
    go_start_idx = features_list[0].shape[0]
    go_end_idx = go_start_idx + features_list[1].shape[0]



    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type in [1, 5]:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type in [2, 4]:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = torch.LongTensor(np.vstack((np.arange(dim), np.arange(dim))))
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = torch.LongTensor(np.vstack((np.arange(dim), np.arange(dim))))
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])

    import scipy.sparse as sp

    # 从 dl.links['data'] 里挑选要用的关系矩阵，排除结构相似性边（关系类型 2）
    selected_mats = [mat for k, mat in dl.links['data'].items() if k != 2]

    # 把选中的各类边矩阵相加得到邻接
    if len(selected_mats) == 0:
        adj_used = sp.csr_matrix((dl.nodes['total'], dl.nodes['total']), dtype=np.float32)
    elif len(selected_mats) == 1:
        adj_used = selected_mats[0]
    else:
        adj_used = reduce(lambda a, b: a + b, selected_mats)

    # g = dgl.DGLGraph(adjM + adjM.T)
    g = dgl.DGLGraph(adj_used + adj_used.T)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(dl.links_test['data'])
    first_flag = True

    for test_edge_type in dl.links_test['data'].keys():
        train_pos, valid_pos = dl.get_train_valid_pos()
        train_pos = train_pos[test_edge_type]
        valid_pos = valid_pos[test_edge_type]

        heads = [args.num_heads] * args.num_layers + [args.num_heads]
        net = myGAT(
            g, args.edge_feats, len(dl.links['count']) * 2 + 1,
            in_dims, args.hidden_dim, args.hidden_dim,
            args.num_layers, heads, F.elu,
            args.dropout, args.dropout, args.slope,
            args.residual, args.residual_att, decode=args.decoder)
        net.to(device)

        # —— 正样本查表（CPU 侧，稀疏矩阵） ——
        import scipy.sparse as sp

        go_start_idx = features_list[0].shape[0]
        go_end_idx = go_start_idx + features_list[1].shape[0]
        num_proteins = go_start_idx
        num_go_terms = go_end_idx - go_start_idx

        # 仅用训练集即可；若担心跨划分“假负”，可把 valid/test 也并进去（见下方注释）
        rows = np.array(train_pos[0])  # 绝对蛋白索引
        cols = np.array(train_pos[1]) - go_start_idx  # GO 转为 0..num_go_terms-1
        data = np.ones_like(rows, dtype=np.uint8)
        pos_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_proteins, num_go_terms))

        # （可选更稳妥）并上 valid/test 的正样本：
        # rows = np.concatenate([rows, np.array(valid_pos[0]), np.array(test_pos[0])])
        # cols = np.concatenate([cols, np.array(valid_pos[1]) - go_start_idx, np.array(test_pos[1]) - go_start_idx])
        # data = np.ones_like(rows, dtype=np.uint8)
        # pos_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_proteins, num_go_terms))

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=f'checkpoint/checkpoint_{args.dataset}_{args.num_layers}.pt')

        # loss_func = nn.BCELoss()
        if args.pos_weight <= 0:
            pos_weight = torch.tensor([float(args.neg_ratio)], device=device)
        else:
            pos_weight = torch.tensor([args.pos_weight], device=device)
        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(args.epoch):
            pos_head, pos_tail = np.array(train_pos[0]), np.array(train_pos[1])
            train_idx = np.arange(len(pos_head))
            np.random.shuffle(train_idx)

            for step, start in enumerate(range(0, len(pos_head), args.batch_size)):
                net.train()

                ph = pos_head[train_idx[start:start + args.batch_size]]
                pt = pos_tail[train_idx[start:start + args.batch_size]]

                # 正样本
                pos_left = torch.LongTensor(ph).to(device)
                pos_right = torch.LongTensor(pt).to(device)
                pos_labels = torch.ones(len(pos_left), device=device)

                neg_left = pos_left.repeat_interleave(20)  # 每个正样本复制 5 次
                neg_right = torch.randint(go_start_idx, go_end_idx, (len(neg_left),), dtype=torch.long).to(device)

                # === 半难负样本：用当前模型在候选池打分，挑高分但非正的 GO，替换部分随机负样本 ===
                if args.hardneg and len(pos_left) > 0:
                    hard_per_pos = max(1, int(args.neg_ratio * args.hardneg_frac))  # 每个正样本替换多少个难负
                    K = args.hardneg_candK  # 候选池大小

                    # 暂时切 eval + no_grad，避免 dropout 干扰且不回传梯度
                    was_training = net.training
                    net.eval()
                    with torch.no_grad():
                        for i, pid in enumerate(pos_left.tolist()):
                            # 候选 GO（随机采 K 个）：注意要过滤掉真阳性
                            cand = torch.randint(go_start_idx, go_end_idx, (K,), device=device)

                            # 过滤：去掉 (pid, cand) 中属于真阳性的条目
                            # 使用你上面构建的 pos_matrix（protein x go_terms）, 列索引需要转为 [0..num_go_terms-1]
                            rows = np.full(K, pid, dtype=np.intp)
                            cols = (cand - int(go_start_idx)).detach().cpu().numpy().astype(np.intp)
                            mask_pos = pos_matrix[rows, cols].A1.astype(bool)
                            if mask_pos.any():
                                keep = torch.from_numpy(~mask_pos).to(device)
                                cand = cand[keep]
                                if cand.numel() == 0:
                                    continue  # 这一轮候选全是正的，跳过

                            # 对候选打分，挑 top-M 作为难负
                            left_all = torch.full((cand.numel(),), pid, dtype=torch.long, device=device)
                            mid_all = torch.zeros_like(left_all)
                            score = net(features_list, e_feat, left_all, cand, mid_all).view(-1)

                            M = min(hard_per_pos, cand.numel())
                            if M == 0:
                                continue
                            top_idx = torch.topk(score, M).indices
                            hard_neg = cand[top_idx]  # [M]

                            # 把这 M 个难负，替换到该蛋白对应的 neg_right 段的前 M 个位置
                            # 该蛋白在 neg_right 中对应的切片：
                            start = i * args.neg_ratio
                            end = start + args.neg_ratio
                            replace_end = start + M
                            neg_right[start:replace_end] = hard_neg

                    # 切回训练模式（如果之前是训练）
                    if was_training:
                        net.train()

                # 1) 行索引：循环外取一次“可写 + 一维 + 整型 + 连续”的副本
                rows = np.ascontiguousarray(neg_left.detach().cpu().numpy(), dtype=np.intp).ravel().copy()

                max_retry = 3
                for _ in range(max_retry):
                    # 2) 列索引：每轮基于当前 neg_right 取一次“可写”的副本
                    cols = np.ascontiguousarray(neg_right.detach().cpu().numpy(), dtype=np.intp).ravel().copy()
                    cols = cols - int(go_start_idx)  # 转为 0..num_go_terms-1

                    hit = pos_matrix[rows, cols].A1.astype(bool)  # ✔ 不会再触发 WRITEABLE 报错
                    if not hit.any():
                        break

                    reidx = torch.from_numpy(np.nonzero(hit)[0]).to(neg_right.device)
                    neg_right[reidx] = torch.randint(
                        go_start_idx, go_end_idx, (len(reidx),),
                        dtype=torch.long, device=neg_right.device
                    )

                # 确保负样本不和正样本重复（optional 简略处理）
                # 注意：真实项目可加 exclude set 做精细处理

                neg_labels = torch.zeros(len(neg_left), device=device)

                # 拼接正负样本
                left = torch.cat([pos_left, neg_left], dim=0)
                right = torch.cat([pos_right, neg_right], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)

                # 中间节点（如不使用中介关系可保持不变）
                mid = torch.zeros(len(left), dtype=torch.long).to(device)

                # 损失函数（使用 BCEWithLogitsLoss，包含 pos_weight）
                logits = net(features_list, e_feat, left, right, mid)
                train_loss = loss_func(logits.view(-1), labels.view(-1))

                # 加入 factor regularization
                node_repr = net.get_node_representation(features_list, e_feat)  # [N, D] 所有节点的表示
                target_node_repr = node_repr[left]  # [B, D]

                factor_K = args.factor_K  # 从 args 中获取 factor_K
                factor_tensor = factorize(target_node_repr, factor_K)  # [B, K, D//K]

                compact_loss = factor_compactness(factor_tensor)
                irrelevance_loss = factor_irrelevance(factor_tensor)

                lambda_c = args.lambda_c  # 从 args 中获取 lambda_c
                lambda_i = args.lambda_i  # 从 args 中获取 lambda_i，保证它们一致

                train_loss += lambda_c * compact_loss + lambda_i * irrelevance_loss

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            net.eval()
            val_losses = []

            with torch.no_grad():
                # 重新计算 GO 术语范围
                go_start_idx = features_list[0].shape[0]
                go_end_idx = go_start_idx + features_list[1].shape[0]
                num_go_terms = go_end_idx - go_start_idx

                # 在循环外，先转换为 numpy 数组
                valid_head = np.array(valid_pos[0])
                valid_tail = np.array(valid_pos[1])

                # 对每个验证蛋白质做全集 GO 打分
                for pid in np.unique(valid_head):
                    # 1. 构造 pid 与全集 GO 的所有配对
                    left = torch.full((num_go_terms,), pid, dtype=torch.long).to(device)
                    right = torch.arange(go_start_idx, go_end_idx, dtype=torch.long).to(device)
                    mid = torch.zeros_like(left).to(device)

                    # 2. 构造标签：正样本 GO→1，其余→0
                    labels = torch.zeros(num_go_terms, device=device)
                    mask = (valid_head == pid)
                    ts = valid_tail[mask]
                    labels[ts - go_start_idx] = 1.0

                    # 3. 前向 + 计算 BCEWithLogitsLoss
                    logits = net(features_list, e_feat, left, right, mid).view(-1)
                    loss_i = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                    val_losses.append(loss_i.item())

            # 4. 求平均 val_loss 并打印
            val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch:03d} Train_Loss: {train_loss.item():.4f} | Val_Loss: {val_loss:.4f}")

            # 5. 早停依据 val_loss
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                break

        net.load_state_dict(torch.load(f'checkpoint/checkpoint_{args.dataset}_{args.num_layers}.pt'))
        net.eval()
        # # === 保存用于预测的 GO term 顺序 ===
        # go_start_idx = features_list[0].shape[0]
        # go_end_idx = go_start_idx + features_list[1].shape[0]
        #
        # terms = [str(i) for i in range(go_start_idx, go_end_idx)]
        # terms_path = f"./prepairs0.7/{args.dataset}/terms.txt"
        # os.makedirs(os.path.dirname(terms_path), exist_ok=True)
        # with open(terms_path, "w") as f:
        #     for tid in terms:
        #         f.write(f"{tid}\n")
        # print(f"✅ 已保存用于评估的 GO term 顺序至: {terms_path}")

        with torch.no_grad():
            net.load_state_dict(torch.load(f'checkpoint/checkpoint_{args.dataset}_{args.num_layers}.pt'))
            net.eval()

            with torch.no_grad():
                # test_proteins = np.unique(dl.links_test['data'][test_edge_type][0])
                # num_go_terms = go_end_idx - go_start_idx
                #
                # y_true_list = []
                # y_pred_list = []
                #
                # for pid in test_proteins:
                #     left = torch.full((num_go_terms,), pid, dtype=torch.long).to(device)
                #     right = torch.arange(go_start_idx, go_end_idx, dtype=torch.long).to(device)
                #     mid = torch.zeros_like(left).to(device)
                #
                #     labels = torch.zeros(num_go_terms, dtype=torch.float32).to(device)
                #     for h, t in zip(*dl.links_test['data'][test_edge_type]):
                #         if h == pid:
                #             labels[t - go_start_idx] = 1.0
                #
                #     logits = net(features_list, e_feat, left, right, mid)
                #     probs = torch.sigmoid(logits)
                #
                #     y_true_list.append(labels.cpu().numpy())
                #     y_pred_list.append(probs.cpu().numpy())
                # 正确做法：直接把稀疏矩阵转为PG边对
                test_mat = dl.links_test['data'][test_edge_type]
                test_head, test_tail = test_mat.nonzero()
                test_proteins = np.unique(test_head)
                num_go_terms = go_end_idx - go_start_idx

                y_true_list = []
                y_pred_list = []

                for pid in test_proteins:
                    left = torch.full((num_go_terms,), int(pid), dtype=torch.long).to(device)
                    right = torch.arange(go_start_idx, go_end_idx, dtype=torch.long).to(device)
                    mid = torch.zeros_like(left).to(device)

                    labels = torch.zeros(num_go_terms, dtype=torch.float32).to(device)
                    mask = (test_head == pid)
                    ts = test_tail[mask]
                    labels[ts - go_start_idx] = 1.0

                    logits = net(features_list, e_feat, left, right, mid)
                    probs = torch.sigmoid(logits)

                    y_true_list.append(labels.cpu().numpy())
                    y_pred_list.append(probs.cpu().numpy())

                from scripts.Evaluation import main as evaluate_all

                y_true = np.array(y_true_list)
                y_pred = np.array(y_pred_list)

                res = evaluate_all(y_true, y_pred)

                # print("✅ Test Smin:", res['smin'])
                # print("✅ Test AUC:", res['auc'])
                # print("✅ Test Fmax:", res['fmax'])
                # print("✅ Test micro/macro F1:", res['micro_f1'], res['macro_f1'])

            for k in res:
                res_random[k] += res[k]

    for k in res_2hop:
        res_2hop[k] /= total
    for k in res_random:
        res_random[k] /= total
    print("✅ 2-hop 指标:", res_2hop)
    print("✅ 全部指标:", res_random)

    final_rec = {
        "phase": "end",
        "run_id": args.run_id,  # 用于和 start 关联
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "neg_ratio": args.neg_ratio,
        "pos_weight": args.pos_weight,
        "metrics": res,
    }
    
    append_json_line(log_file_path, final_rec)


def append_json_line(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=4, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=100, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=2e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--decoder', type=str, default='dot')#dot  distmult  bilinear
    ap.add_argument('--residual-att', type=float, default=0.)
    ap.add_argument('--residual', type=bool, default=False)
    ap.add_argument('--run', type=int, default=1)
    ap.add_argument('--pos-weight', type=int, default=15,
                    help='正样本权重，<=0 时自动等于 neg_ratio')
    ap.add_argument('--neg-ratio', type=int, default=15, help='每个正样本配多少负样本')
    ap.add_argument('--hardneg', action='store_true', help='启用半难负样本')
    ap.add_argument('--hardneg-frac', type=float, default=0.5, help='每个正样本负样本中有多少比例用难负来替换，0~1')
    ap.add_argument('--hardneg-candK', type=int, default=512, help='每个蛋白抽取多少候选 GO 用于打分挑难负')
    ap.add_argument('--run_id', type=str, default='', help='由网格脚本传入，用于关联 start/end')
    # 在 argparse 中添加这几个参数
    ap.add_argument('--factor_K', type=int, default=2, help='Number of factors for factorization')
    ap.add_argument('--lambda_c', type=float, default=1.0, help='Weight for compactness loss')
    ap.add_argument('--lambda_i', type=float, default=1.0, help='Weight for irrelevance loss')

    args = ap.parse_args()
    run_model_DBLP(args)