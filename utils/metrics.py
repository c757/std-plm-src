import torch


def _safe_apply(metric_fn, pred, true):
    if pred.numel() == 0 or true.numel() == 0:
        return float('nan')
    return metric_fn(pred=pred, true=true).item()

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred))

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def MAPE_torch(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))

def MAPE_torch_node(pred, true, mask_value=1e-6):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = pred*mask
        true = true*mask + (1-mask.float())
        count = mask.sum(dim=-1)
    return torch.sum(torch.abs(torch.div((true - pred)*mask, true)),dim=-1)/count


def ACC_torch(pred, true, eps=1e-8):
    # 回归场景的相关系数（Anomaly Correlation Coefficient 风格）
    pred = pred - pred.mean()
    true = true - true.mean()
    denom = torch.sqrt(torch.mean(pred ** 2) * torch.mean(true ** 2)) + eps
    return torch.mean(pred * true) / denom


def cal_metrics(predicts,targets,eval_mask):
    F = targets.shape[-1]

    mae = []
    for f in range(F):
        mask = eval_mask[...,f]
        mae.append(_safe_apply(MAE_torch, predicts[...,f][mask], targets[...,f][mask]))

    rmse = []
    for f in range(F):
        mask = eval_mask[...,f]
        rmse.append(_safe_apply(RMSE_torch, predicts[...,f][mask], targets[...,f][mask]))

    mape = []
    for f in range(F):
        mask = eval_mask[...,f]
        mape.append(_safe_apply(MAPE_torch, predicts[...,f][mask], targets[...,f][mask]))

    acc = []
    for f in range(F):
        mask = eval_mask[...,f]
        pred_f = predicts[...,f][mask]
        true_f = targets[...,f][mask]
        if pred_f.numel() == 0:
            acc.append(float('nan'))
        else:
            acc.append(ACC_torch(pred_f, true_f).item())

    mape_10 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 10)
        mape_10.append(_safe_apply(MAPE_torch, predicts[...,f][mask], targets[...,f][mask]))

    mape_20 = []
    for f in range(F):
        mask = eval_mask[...,f]
        mask = mask & (targets[...,0] >= 20)
        mape_20.append(_safe_apply(MAPE_torch, predicts[...,f][mask], targets[...,f][mask]))  

    return mae,rmse,mape,acc,mape_10,mape_20

def VRMSE_torch(pred_u, pred_v, true_u, true_v, mask_value=None):
    """
    矢量均方根误差：RMSE of vector magnitude error
    pred_u/v, true_u/v: same shape tensors
    """
    if mask_value is not None:
        mag_true = torch.sqrt(true_u ** 2 + true_v ** 2)
        mask = torch.gt(mag_true, mask_value)
        pred_u = torch.masked_select(pred_u, mask)
        pred_v = torch.masked_select(pred_v, mask)
        true_u = torch.masked_select(true_u, mask)
        true_v = torch.masked_select(true_v, mask)
    err_u = pred_u - true_u
    err_v = pred_v - true_v
    return torch.sqrt(torch.mean(err_u ** 2 + err_v ** 2))