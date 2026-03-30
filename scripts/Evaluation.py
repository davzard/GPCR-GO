from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, roc_auc_score, \
    roc_curve
import numpy as np
import json

def auprc(ytrue, ypred):
    """
    根据预测概率计算精确率-召回率曲线，并返回该曲线下的面积（AUPR）
    """
    p, r, _ = precision_recall_curve(ytrue, ypred)
    return auc(r, p)

# 新增：频次估计 IC
def compute_ic_from_ytrue(ytrue, eps=1e-8):
    """
    根据标签矩阵频次估计每个 GO 术语的信息量（IC）。
    IC(term_j) = -log((freq_j + eps) / (N + eps))
    """
    ytrue = np.asarray(ytrue)
    N = ytrue.shape[0]
    freq = ytrue.sum(axis=0).astype(np.float64)
    ic = -np.log((freq + eps) / (N + eps))
    return ic.astype(np.float32)

# 新增：阈值扫描求 Smin
def smin_from_arrays(ytrue, ypred_prob, ic_vec=None, thresholds=None):
    """
    Smin 计算（CAFA 风格）：
    对每个阈值：二值化 -> 每蛋白 RU=sum(IC of FN), MI=sum(IC of FP)
    对全体蛋白取 RU/MI 的均值 -> S(th)=sqrt(RU^2+MI^2) -> 取最小
    """
    ytrue = np.asarray(ytrue).astype(int)
    ypred_prob = np.asarray(ypred_prob)
    if ic_vec is None:
        ic_vec = compute_ic_from_ytrue(ytrue)
    ic_vec = np.asarray(ic_vec).reshape(-1)

    N, M = ytrue.shape
    if thresholds is None:
        thresholds = [t/100.0 for t in range(1, 101)]

    s_values = []
    for th in thresholds:
        ypred_bin = (ypred_prob > th).astype(int)
        RU_list, MI_list = [], []
        for i in range(N):
            true_idx = np.where(ytrue[i] == 1)[0]
            pred_idx = np.where(ypred_bin[i] == 1)[0]
            miss_idx = np.setdiff1d(true_idx, pred_idx, assume_unique=False)  # FN
            extra_idx = np.setdiff1d(pred_idx, true_idx, assume_unique=False)  # FP
            ru = ic_vec[miss_idx].sum() if miss_idx.size > 0 else 0.0
            mi = ic_vec[extra_idx].sum() if extra_idx.size > 0 else 0.0
            RU_list.append(ru); MI_list.append(mi)
        RU = float(np.mean(RU_list)) if RU_list else 0.0
        MI = float(np.mean(MI_list)) if MI_list else 0.0
        s_values.append((RU * RU + MI * MI) ** 0.5)
    return float(np.min(s_values)) if s_values else 0.0


def main(ytrue1, ypred1):
    """
    评价指标计算函数
    输入：
        ytrue1: 原始真实标签（一维数组或列表，长度为样本数）
        ypred1: 原始预测概率（一维数组或列表，长度为样本数）
    返回：
        一个字典，包含以下指标：
         - fmax: 遍历阈值得到的最大 F1 分数
         - aupr: AUPR
         - micro_f1: 微平均 F1
         - macro_f1: 宏平均 F1
         - precision: 微平均精确率
         - recall: 微平均召回率
         - auc: 利用 ROC 曲线计算得到的 AUC
    """
    # 将输入转换为矩阵格式（模拟原作者做法）
    ytrue1 = np.mat(ytrue1)
    ypred1 = np.mat(ypred1)
    fmax = 0
    ytrue = []
    ypred = []

    # 过滤掉没有正标签的样本
    for i in range(len(ytrue1)):
        if np.sum(ytrue1[i]) > 0:
            ytrue.append(ytrue1[i])
            ypred.append(ypred1[i])

    # 转换为 numpy 数组；如果有多余维度则 squeeze 成 (N, M)
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    if ytrue.ndim > 2:
        ytrue = ytrue.squeeze(1)
        ypred = ypred.squeeze(1)

    # 遍历阈值 0.01 ~ 1.00 计算 F1 分数，获取 Fmax
    for t in range(1, 101):
        thres = t / 100.0
        thres_value = np.ones((len(ytrue), ytrue.shape[1]), dtype=np.float32) * thres
        pred_values = (ypred > thres_value).astype(int)
        tp_matrix = pred_values * ytrue

        tp = np.sum(tp_matrix, axis=1, dtype=np.int32)
        tpfp = np.sum(pred_values, axis=1)
        tpfn = np.sum(ytrue, axis=1)

        avg_pr = []
        for i in range(len(tp)):
            if tpfp[i] != 0:
                avg_pr.append(tp[i] / float(tpfp[i]))
        if len(avg_pr) == 0:
            continue
        avgpr = np.mean(avg_pr)
        avgrc = np.mean(tp / tpfn)
        if avgpr == 0 and avgrc == 0:
            f1 = 0
        else:
            f1 = 2 * avgpr * avgrc / (avgpr + avgrc)
        fmax = max(fmax, f1)

    # 计算 AUPR：对过滤后的所有样本展平后计算
    aupr_value = auprc(np.array(ytrue).flatten(), np.array(ypred).flatten())

    # 固定阈值（0.5）下二值化，用于计算其它指标
    ytrue_arr = np.array(ytrue)
    ypred_arr = np.array(ypred)
    ypred_bin = (ypred_arr > 0.5).astype(int)

    micro_f1 = f1_score(ytrue_arr, ypred_bin, average='micro', zero_division=0)
    macro_f11 = f1_score(ytrue_arr, ypred_bin, average='macro', zero_division=0)
    prec = precision_score(ytrue_arr, ypred_bin, average='micro', zero_division=0)
    rec = recall_score(ytrue_arr, ypred_bin, average='micro', zero_division=0)
    valid_columns = np.sum(ytrue_arr, axis=0) > 0
    if valid_columns.any():
        macro_f1 = f1_score(ytrue_arr[:, valid_columns], ypred_bin[:, valid_columns], average='macro', zero_division=0)
    else:
        macro_f1 = 0.0

    # 计算 ROC 曲线及 AUC
    ytrue_orig = np.array(ytrue1).flatten()
    ypred_orig = np.array(ypred1).flatten()
    if len(np.unique(ytrue_orig)) < 2:
        auc_value = None
        roc_auc = None
    else:
        fpr, tpr, _ = roc_curve(ytrue_orig, ypred_orig)
        auc_value = auc(fpr, tpr)
        roc_auc = roc_auc_score(ytrue_orig, ypred_orig)

    smin = smin_from_arrays(ytrue_arr, ypred_arr)

    results = {
        "roc_auc": roc_auc,
        'fmax': fmax,
        'aupr': aupr_value,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'macro_f11': macro_f11,
        'precision': prec,
        'recall': rec,
        'auc': auc_value,
        'smin': smin
    }

    return results
