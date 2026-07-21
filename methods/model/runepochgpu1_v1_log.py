import json
import sys
import time
import argparse
import random
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from GNN import myGAT
import dgl
import re
from functools import reduce
import scipy.sparse as sp


PAPER_DEFAULTS = {
    "bp": {"neg_ratio": 15, "pos_weight": 15, "factor_K": 8, "lambda_reg": 0.1},
    "cc": {"neg_ratio": 10, "pos_weight": 15, "factor_K": 2, "lambda_reg": 0.1},
    "mf": {"neg_ratio": 15, "pos_weight": 15, "factor_K": 4, "lambda_reg": 1.0},
}
RELATION_NAMES = {
    0: "protein_go_annotation",
    1: "ppi",
    2: "structural_similarity",
    3: "go_hierarchy",
}
GDN_REGULARIZATION_MODE = "paper_inter_factor_all_nodes_shared_forward"
USE_UNREVIEWED_PROTEINS = True


def add_boolean_flag(parser, enabled_flag, disabled_flag, dest, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(enabled_flag, dest=dest, action="store_true", help=help_text)
    group.add_argument(
        disabled_flag,
        dest=dest,
        action="store_false",
        help=f"Disable {enabled_flag.lstrip('-')}.",
    )
    parser.set_defaults(**{dest: default})

def apply_paper_defaults(args, parser):
    aspect = Path(args.dataset).name.lower()
    if aspect not in PAPER_DEFAULTS:
        parser.error("--dataset must end in /bp, /cc, or /mf")
    defaults = PAPER_DEFAULTS[aspect]
    args.aspect = aspect

    for name in ("neg_ratio", "pos_weight", "factor_K"):
        if getattr(args, name) is None:
            setattr(args, name, defaults[name])

    if args.lambda_reg is not None and (args.lambda_c is not None or args.lambda_i is not None):
        parser.error("use either --lambda or --lambda-c/--lambda-i, not both")
    shared_lambda = defaults["lambda_reg"] if args.lambda_reg is None else args.lambda_reg
    args.lambda_c = shared_lambda if args.lambda_c is None else args.lambda_c
    args.lambda_i = shared_lambda if args.lambda_i is None else args.lambda_i

    representation_dim = args.hidden_dim * (args.num_layers + 2)
    if args.factor_K < 1 or representation_dim % args.factor_K != 0:
        parser.error(
            f"--num-factors must divide the decoder representation dimension ({representation_dim})"
        )
    if args.neg_ratio < 1:
        parser.error("--neg-ratio/Pe must be >= 1")
    if args.pos_weight <= 0:
        parser.error("--pos-weight/Pn must be > 0")
    if not 0.0 <= args.hardneg_frac <= 1.0:
        parser.error("--hardneg-frac/Ph must be in [0, 1]")
    if args.hardneg_candK < 1:
        parser.error("--hardneg-candK must be >= 1")


def experiment_summary(args):
    return {
        "dataset": args.dataset,
        "aspect": args.aspect,
        "neg_ratio_Pe": args.neg_ratio,
        "pos_weight_Pn": args.pos_weight,
        "hard_negative_mining": args.hardneg,
        "hard_negative_fraction_Ph": args.hardneg_frac,
        "hard_negative_candidate_pool": args.hardneg_candK,
        "num_factors_M": args.factor_K,
        "lambda_c": args.lambda_c,
        "lambda_i": args.lambda_i,
        "gdn_regularization_mode": GDN_REGULARIZATION_MODE,
        "use_annotation_edges": args.use_annotation_edges,
        "use_ppi": args.use_ppi,
        "use_structural_similarity": args.use_struct_sim,
        "use_go_hierarchy": args.use_go_hierarchy,
        "use_unreviewed_proteins_SSL": USE_UNREVIEWED_PROTEINS,
        "seed": args.seed,
    }

def safe_path_part(value):
    value = str(value)
    value = value.replace("\\", "_").replace("/", "_")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)

def hardneg_count_per_positive(neg_ratio, hardneg_frac):
    if neg_ratio < 1:
        raise ValueError("--neg-ratio must be >= 1")
    if hardneg_frac < 0.0 or hardneg_frac > 1.0:
        raise ValueError("--hardneg-frac/P_h must be in [0, 1]")
    return min(neg_ratio, max(0, int(neg_ratio * hardneg_frac)))

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

def factorize(h, num_factors):
    B, D = h.shape
    factor_dim = D // num_factors
    return h.view(B, num_factors, factor_dim)

def inter_factor_node_repulsion(factor_tensor):
    """Mean raw cosine similarity between factor pairs of the same node."""
    _, K, _ = factor_tensor.shape
    if K < 2:
        return torch.zeros((), dtype=factor_tensor.dtype, device=factor_tensor.device)

    normalized = F.normalize(factor_tensor, p=2, dim=-1)
    cosine_matrix = torch.bmm(normalized, normalized.transpose(1, 2))
    pair_mask = torch.triu(
        torch.ones(K, K, dtype=torch.bool, device=factor_tensor.device), diagonal=1
    )
    return cosine_matrix[:, pair_mask].mean()

def inter_factor_space_separation(factor_tensor):
    """Negative mean squared distance between factor-space centroids."""
    _, K, _ = factor_tensor.shape
    if K < 2:
        return torch.zeros((), dtype=factor_tensor.dtype, device=factor_tensor.device)

    factor_means = factor_tensor.mean(dim=0)
    mean_differences = factor_means.unsqueeze(1) - factor_means.unsqueeze(0)
    squared_distances = mean_differences.square().sum(dim=-1)
    pair_mask = torch.triu(
        torch.ones(K, K, dtype=torch.bool, device=factor_tensor.device), diagonal=1
    )
    return -squared_distances[pair_mask].mean()

def run_model_DBLP(args):
    if not hasattr(args, 'seed'):
        args.seed = 42
    if not hasattr(args, 'eval_split'):
        args.eval_split = 'test'
    if not hasattr(args, 'use_struct_sim'):
        args.use_struct_sim = False
    if not hasattr(args, 'struct_sim_rid'):
        args.struct_sim_rid = 2
    setup_seed(args.seed)


    log_file_path = getattr(args, 'log_file', './training_logs.jsonl')
    hardneg_per_pos = hardneg_count_per_positive(args.neg_ratio, args.hardneg_frac) if args.hardneg else 0
    effective_hardneg_frac = hardneg_per_pos / float(args.neg_ratio)
    effective_run_id = args.run_id or (
        f"{safe_path_part(args.dataset)}_Pe{args.neg_ratio}_Pn{args.pos_weight}"
        f"_Ph{args.hardneg_frac}_M{args.factor_K}"
        f"_lc{args.lambda_c}_li{args.lambda_i}_seed{args.seed}_gdnPaper"
    )
    start_rec = {
        "phase": "start",
        "run_id": effective_run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "use_struct_sim": args.use_struct_sim,
        "struct_sim_rid": args.struct_sim_rid,
        "use_annotation_edges": args.use_annotation_edges,
        "use_ppi": args.use_ppi,
        "use_go_hierarchy": args.use_go_hierarchy,
        "use_unreviewed": USE_UNREVIEWED_PROTEINS,
        "neg_ratio": args.neg_ratio,
        "pos_weight": args.pos_weight,
        "hardneg_enabled": args.hardneg,
        "hardneg_frac": args.hardneg_frac,
        "hardneg_per_pos": hardneg_per_pos,
        "effective_hardneg_frac": effective_hardneg_frac,
        "hardneg_candK": args.hardneg_candK,
        "factor_K": args.factor_K,
        "lambda_c": args.lambda_c,
        "lambda_i": args.lambda_i,
        "gdn_regularization_mode": GDN_REGULARIZATION_MODE,
        "checkpoint_dir": args.checkpoint_dir,
        "save_mode": args.save_mode,
    }
    append_json_line(log_file_path, start_rec)


    feats_type = args.feats_type
    features_list, _, dl = load_data(args.dataset)
    print(f"validation_source={dl.validation_source}")
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

    relation_enabled = {
        0: args.use_annotation_edges,
        1: args.use_ppi,
        args.struct_sim_rid: args.use_struct_sim,
        3: args.use_go_hierarchy,
    }
    selected_relation_ids = [
        relation_id
        for relation_id in dl.links['data'].keys()
        if relation_enabled.get(relation_id, True)
    ]
    selected_mats = [dl.links['data'][relation_id] for relation_id in selected_relation_ids]
    print(
        "selected relations: "
        + ", ".join(
            f"{relation_id}:{RELATION_NAMES.get(relation_id, 'unknown')}"
            for relation_id in selected_relation_ids
        )
        + f"; use_unreviewed={USE_UNREVIEWED_PROTEINS}"
    )


    if len(selected_mats) == 0:
        adj_used = sp.csr_matrix((dl.nodes['total'], dl.nodes['total']), dtype=np.float32)
    elif len(selected_mats) == 1:
        adj_used = selected_mats[0]
    else:
        adj_used = reduce(lambda a, b: a + b, selected_mats)


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

    from scripts.Evaluation import load_ic_vector

    go_start_idx = features_list[0].shape[0]
    go_end_idx = go_start_idx + features_list[1].shape[0]
    ic_path = Path(dl.path) / "ic_dict.json"
    if not ic_path.is_file():
        raise FileNotFoundError(f"Fixed IC file not found: {ic_path}")
    ic_vec = load_ic_vector(ic_path, range(go_start_idx, go_end_idx))

    res_random = defaultdict(float)
    res_validation = defaultdict(float)
    total = len(dl.links_test['data'])
    run_id = effective_run_id
    model_root = os.path.join(args.checkpoint_dir, run_id)
    os.makedirs(model_root, exist_ok=True)
    best_model_paths = []

    for test_edge_type in dl.links_test['data'].keys():
        edge_tag = safe_path_part(test_edge_type)
        best_model_path = os.path.join(model_root, f"edge{edge_tag}_best.pt")
        best_model_paths.append(best_model_path)
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


        import scipy.sparse as sp

        go_start_idx = features_list[0].shape[0]
        go_end_idx = go_start_idx + features_list[1].shape[0]
        num_proteins = go_start_idx
        num_go_terms = go_end_idx - go_start_idx


        rows = np.array(train_pos[0])
        cols = np.array(train_pos[1]) - go_start_idx
        data = np.ones_like(rows, dtype=np.uint8)
        pos_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_proteins, num_go_terms))

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path=best_model_path)


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


                pos_left = torch.LongTensor(ph).to(device)
                pos_right = torch.LongTensor(pt).to(device)
                pos_labels = torch.ones(len(pos_left), device=device)

                neg_left = pos_left.repeat_interleave(args.neg_ratio)
                neg_right = torch.randint(go_start_idx, go_end_idx, (len(neg_left),), dtype=torch.long).to(device)

                if args.hardneg and hardneg_per_pos > 0 and len(pos_left) > 0:
                    hard_per_pos = hardneg_per_pos
                    K = args.hardneg_candK

                    was_training = net.training
                    net.eval()
                    with torch.no_grad():
                        for i, pid in enumerate(pos_left.tolist()):
                            cand = torch.randint(go_start_idx, go_end_idx, (K,), device=device)
                            rows = np.full(K, pid, dtype=np.intp)
                            cols = (cand - int(go_start_idx)).detach().cpu().numpy().astype(np.intp)
                            mask_pos = pos_matrix[rows, cols].A1.astype(bool)
                            if mask_pos.any():
                                keep = torch.from_numpy(~mask_pos).to(device)
                                cand = cand[keep]
                                if cand.numel() == 0:
                                    continue

                            left_all = torch.full((cand.numel(),), pid, dtype=torch.long, device=device)
                            mid_all = torch.zeros_like(left_all)
                            score = net(features_list, e_feat, left_all, cand, mid_all).view(-1)

                            M = min(hard_per_pos, cand.numel())
                            if M == 0:
                                continue
                            top_idx = torch.topk(score, M).indices
                            hard_neg = cand[top_idx]

                            slice_start = i * args.neg_ratio
                            replace_end = slice_start + M
                            neg_right[slice_start:replace_end] = hard_neg


                    if was_training:
                        net.train()

                rows = np.ascontiguousarray(neg_left.detach().cpu().numpy(), dtype=np.intp).ravel().copy()
                max_retry = 3
                for _ in range(max_retry):
                    cols = np.ascontiguousarray(neg_right.detach().cpu().numpy(), dtype=np.intp).ravel().copy()
                    cols = cols - int(go_start_idx)

                    hit = pos_matrix[rows, cols].A1.astype(bool)
                    if not hit.any():
                        break

                    reidx = torch.from_numpy(np.nonzero(hit)[0]).to(neg_right.device)
                    neg_right[reidx] = torch.randint(
                        go_start_idx, go_end_idx, (len(reidx),),
                        dtype=torch.long, device=neg_right.device
                    )
                neg_labels = torch.zeros(len(neg_left), device=device)
                left = torch.cat([pos_left, neg_left], dim=0)
                right = torch.cat([pos_right, neg_right], dim=0)
                labels = torch.cat([pos_labels, neg_labels], dim=0)
                mid = torch.zeros(len(left), dtype=torch.long).to(device)
                logits, node_repr = net.forward_with_representation(
                    features_list, e_feat, left, right, mid
                )
                train_loss = loss_func(logits.view(-1), labels.view(-1))
                factor_tensor = factorize(node_repr, args.factor_K)
                node_repulsion_loss = inter_factor_node_repulsion(factor_tensor)
                space_separation_loss = inter_factor_space_separation(factor_tensor)
                train_loss += (
                    args.lambda_c * node_repulsion_loss
                    + args.lambda_i * space_separation_loss
                )

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            net.eval()
            val_losses = []

            with torch.no_grad():
                go_start_idx = features_list[0].shape[0]
                go_end_idx = go_start_idx + features_list[1].shape[0]
                num_go_terms = go_end_idx - go_start_idx
                valid_head = np.array(valid_pos[0])
                valid_tail = np.array(valid_pos[1])
                for pid in np.unique(valid_head):

                    left = torch.full((num_go_terms,), pid, dtype=torch.long).to(device)
                    right = torch.arange(go_start_idx, go_end_idx, dtype=torch.long).to(device)
                    mid = torch.zeros_like(left).to(device)
                    labels = torch.zeros(num_go_terms, device=device)
                    mask = (valid_head == pid)
                    ts = valid_tail[mask]
                    labels[ts - go_start_idx] = 1.0
                    logits = net(features_list, e_feat, left, right, mid).view(-1)
                    loss_i = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
                    val_losses.append(loss_i.item())

            val_loss = sum(val_losses) / len(val_losses)
            print(f"Epoch {epoch:03d} Train_Loss: {train_loss.item():.4f} | Val_Loss: {val_loss:.4f}")
            early_stopping(val_loss, net)
            if args.save_mode == "all":
                epoch_model_path = os.path.join(
                    model_root,
                    f"edge{edge_tag}_epoch{epoch:04d}_valloss{val_loss:.6f}.pt"
                )
                torch.save(net.state_dict(), epoch_model_path)
            if early_stopping.early_stop:
                break

        net.load_state_dict(torch.load(best_model_path, map_location=device))
        net.eval()

        def evaluate_full_go(head_arr, tail_arr):
            head_arr = np.array(head_arr)
            tail_arr = np.array(tail_arr)
            eval_proteins = np.unique(head_arr)
            num_go_terms = go_end_idx - go_start_idx

            y_true_list = []
            y_pred_list = []
            with torch.no_grad():
                for pid in eval_proteins:
                    left = torch.full((num_go_terms,), int(pid), dtype=torch.long).to(device)
                    right = torch.arange(go_start_idx, go_end_idx, dtype=torch.long).to(device)
                    mid = torch.zeros_like(left).to(device)

                    labels = torch.zeros(num_go_terms, dtype=torch.float32).to(device)
                    mask = (head_arr == pid)
                    ts = tail_arr[mask]
                    labels[ts - go_start_idx] = 1.0

                    logits = net(features_list, e_feat, left, right, mid)
                    probs = torch.sigmoid(logits)

                    y_true_list.append(labels.cpu().numpy())
                    y_pred_list.append(probs.cpu().numpy())

            from scripts.Evaluation import main as evaluate_all

            y_true = np.array(y_true_list)
            y_pred = np.array(y_pred_list)
            return evaluate_all(y_true, y_pred, ic_vec=ic_vec)

        if args.eval_split in ('validation', 'both'):
            valid_res = evaluate_full_go(np.array(valid_pos[0]), np.array(valid_pos[1]))
            for k, v in valid_res.items():
                res_validation[k] += v

        if args.eval_split in ('test', 'both'):
            test_mat = dl.links_test['data'][test_edge_type]
            test_head, test_tail = test_mat.nonzero()
            test_res = evaluate_full_go(test_head, test_tail)
            for k, v in test_res.items():
                res_random[k] += v

    for k in res_validation:
        res_validation[k] /= total
    for k in res_random:
        res_random[k] /= total
    validation_metrics = dict(res_validation)
    test_metrics = dict(res_random)
    primary_metrics = test_metrics if args.eval_split in ('test', 'both') else validation_metrics
    print("✅ validation 指标:", validation_metrics)
    print("✅ test 指标:", test_metrics)

    final_rec = {
        "phase": "end",
        "run_id": effective_run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "eval_split": args.eval_split,
        "use_struct_sim": args.use_struct_sim,
        "struct_sim_rid": args.struct_sim_rid,
        "use_annotation_edges": args.use_annotation_edges,
        "use_ppi": args.use_ppi,
        "use_go_hierarchy": args.use_go_hierarchy,
        "use_unreviewed": USE_UNREVIEWED_PROTEINS,
        "neg_ratio": args.neg_ratio,
        "pos_weight": args.pos_weight,
        "hardneg_enabled": args.hardneg,
        "hardneg_frac": args.hardneg_frac,
        "hardneg_per_pos": hardneg_per_pos,
        "effective_hardneg_frac": effective_hardneg_frac,
        "metrics": primary_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_source": dl.validation_source,
        "best_model_paths": best_model_paths,
    }

    append_json_line(log_file_path, final_rec)


def append_json_line(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train GPCR-GO on one GO branch. Paper defaults are selected from the "
            "bp/cc/mf suffix of --dataset and can be overridden by CLI parameters."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset path below data/, e.g. reviewed6/bp",
    )
    parser.add_argument("--feats-type", type=int, choices=range(6), default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--slope", type=float, default=0.01)
    parser.add_argument("--edge-feats", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decoder", choices=("dot", "distmult", "bilinear"), default="dot")
    parser.add_argument("--residual-att", type=float, default=0.0)
    add_boolean_flag(
        parser, "--residual", "--no-residual", "residual", False,
        "Enable residual connections.",
    )

    parser.add_argument("--neg-ratio", "--Pe", dest="neg_ratio", type=int, default=None)
    parser.add_argument("--pos-weight", "--Pn", dest="pos_weight", type=float, default=None)
    parser.add_argument("--hardneg-frac", "--Ph", dest="hardneg_frac", type=float, default=0.5)
    parser.add_argument("--hardneg-candK", type=int, default=512)
    parser.add_argument(
        "--factor-K", "--factor_K", "--num-factors", "--M",
        dest="factor_K", type=int, default=None,
    )
    parser.add_argument(
        "--lambda", dest="lambda_reg", type=float, default=None,
        help="Shared paper regularization weight for both GDN losses.",
    )
    parser.add_argument("--lambda-c", "--lambda_c", dest="lambda_c", type=float, default=None)
    parser.add_argument("--lambda-i", "--lambda_i", dest="lambda_i", type=float, default=None)

    add_boolean_flag(
        parser, "--hardneg", "--no-hardneg", "hardneg", True,
        "Enable online hard negative mining (HNM).",
    )
    add_boolean_flag(
        parser, "--use-annotation-edges", "--no-annotation-edges",
        "use_annotation_edges", True, "Use training protein-GO edges in message passing.",
    )
    add_boolean_flag(
        parser, "--use-ppi", "--no-ppi", "use_ppi", True,
        "Use PPI relation edges.",
    )
    add_boolean_flag(
        parser, "--use-struct-sim", "--no-struct-sim", "use_struct_sim", True,
        "Use structural-similarity edges (SSE).",
    )
    add_boolean_flag(
        parser, "--use-go-hierarchy", "--no-go-hierarchy", "use_go_hierarchy", True,
        "Use GO hierarchy edges.",
    )
    parser.add_argument("--struct-sim-rid", type=int, default=2)

    parser.add_argument("--run-id", "--run_id", dest="run_id", default="")
    parser.add_argument("--log-file", default="./training_logs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-split", choices=("validation", "test", "both"), default="test"
    )
    parser.add_argument(
        "--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir",
        default="checkpoint/paper2",
    )
    parser.add_argument(
        "--save-mode", "--save_mode", dest="save_mode",
        choices=("best", "all"), default="best",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print parameters without loading data or training.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_paper_defaults(args, parser)
    print(json.dumps(experiment_summary(args), indent=2, sort_keys=True))
    if args.dry_run:
        return
    os.chdir(SCRIPT_DIR)
    run_model_DBLP(args)


if __name__ == "__main__":
    main()
