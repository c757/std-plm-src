import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import os
from utils.utils import get_time_str, check_dir, draw_loss_line, draw_mape_node, get_randmask, get_block_mask, cal_shortest_path_length
from logger import getlogger
from model.model import STALLM_MIMO
from model.llm import Phi2, GPT2, LLAMA3, Transformer, QWEN
from utils.metrics import MAE_torch, RMSE_torch, MAPE_torch, MAPE_torch_node, cal_metrics
from utils.argsinit import InitArgs
import copy
from torch.optim.lr_scheduler import ExponentialLR
try:
    import nni
except ImportError:
    nni = None
import random
import string
import json
import csv
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from utils.ocean_dataloader import get_ocean_dataloaders, load_ocean_laplacian_embeddings

# 自动检测计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"当前使用的计算设备: {device}")

random_str = lambda : ''.join(random.sample(string.ascii_letters + string.digits, 6))

COMPONENT_NAMES = {
    'flow': ['flow_u', 'flow_v'],
    'wave': ['wave_h', 'wave_dir_sin', 'wave_dir_cos', 'wave_period'],
    'wind': ['wind_u', 'wind_v'],
}

MODALITY_ORDER = ['flow', 'wave', 'wind']

# Per-component physical units (used for MAE/RMSE). Empty string means unitless.
COMPONENT_UNITS = {
    'flow': ['m/s', 'm/s'],
    'wave': ['m', '', '', 's'],
    'wind': ['m/s', 'm/s'],
}


def _fmt_float(v, ndigits=4):
    try:
        if v is None:
            return 'nan'
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return 'nan'


def _format_named_metric(values, var_name, metric_name):
    # Resolve names and units for each channel
    names = COMPONENT_NAMES.get(var_name, [f"{var_name}_{i}" for i in range(len(values))])
    units = COMPONENT_UNITS.get(var_name, ["" for _ in range(len(values))])
    if len(names) < len(values):
        names += [f"{var_name}_{i}" for i in range(len(names), len(values))]
    if len(units) < len(values):
        units += ["" for _ in range(len(units), len(values))]

    pairs = []
    for i in range(len(values)):
        val = values[i]
        unit = units[i]

        # For MAPE show percent
        if metric_name.upper().startswith('MAPE'):
            disp = _fmt_float(float(val) * 100.0)
            suffix = '%'
        elif metric_name.upper().startswith('ACC'):
            # ACC is dimensionless (correlation-like)
            disp = _fmt_float(val)
            suffix = ''
        else:
            disp = _fmt_float(val)
            suffix = f" {unit}" if unit else ''

        pairs.append(f"{names[i]}={disp}{suffix}")

    return f"{metric_name}[{', '.join(pairs)}]"


def _init_fusion_stats():
    return {tgt: {src: 0.0 for src in MODALITY_ORDER} for tgt in MODALITY_ORDER}, 0


def _accumulate_fusion_stats(stats, weight_dict):
    stat_map, count = stats
    if not weight_dict:
        return stat_map, count

    for tgt in MODALITY_ORDER:
        w = weight_dict.get(tgt, None)
        if w is None:
            continue
        # w: (B, S, 3)
        w_mean = w.detach().mean(dim=(0, 1)).cpu().tolist()
        for i, src in enumerate(MODALITY_ORDER):
            stat_map[tgt][src] += float(w_mean[i])
    return stat_map, count + 1


def _finalize_fusion_stats(stats):
    stat_map, count = stats
    if count == 0:
        return {tgt: {src: float('nan') for src in MODALITY_ORDER} for tgt in MODALITY_ORDER}
    out = {}
    for tgt in MODALITY_ORDER:
        out[tgt] = {}
        for src in MODALITY_ORDER:
            out[tgt][src] = stat_map[tgt][src] / count
    return out


def _format_fusion_stats(fusion_stats):
    lines = []
    for tgt in MODALITY_ORDER:
        items = [f"{src}={_fmt_float(fusion_stats[tgt][src])}" for src in MODALITY_ORDER]
        lines.append(f"{tgt}<=({', '.join(items)})")
    return ' | '.join(lines)


def _unpack_model_output(model_output):
    if isinstance(model_output, tuple):
        if len(model_output) == 3:
            return model_output
        if len(model_output) == 2:
            pred, other_loss = model_output
            return pred, other_loss, {}
    return model_output, [], {}


def _init_training_history(chosen_vars):
    history = {
        'train_loss': {'x': [], 'y': []},
        'val_loss': {'x': [], 'y': []},
        'fusion': {
            split: {
                tgt: {src: {'x': [], 'y': []} for src in MODALITY_ORDER}
                for tgt in MODALITY_ORDER
            }
            for split in ['train', 'val', 'test', 'final']
        },
        'test_metrics': {
            var: {
                metric: []  # list of {'epoch': int, 'values': [..]}
                for metric in ['mae', 'rmse', 'mape', 'acc']
            }
            for var in chosen_vars
        }
    }
    return history


def _record_fusion_history(history, split, epoch, fusion_stats):
    if fusion_stats is None:
        return
    for tgt in MODALITY_ORDER:
        for src in MODALITY_ORDER:
            v = fusion_stats.get(tgt, {}).get(src, float('nan'))
            history['fusion'][split][tgt][src]['x'].append(epoch)
            history['fusion'][split][tgt][src]['y'].append(float(v))


def _record_test_metrics_history(history, epoch, results):
    for var, metrics in results.items():
        mae, rmse, mape, acc = metrics
        history['test_metrics'][var]['mae'].append({'epoch': int(epoch), 'values': [float(x) for x in mae]})
        history['test_metrics'][var]['rmse'].append({'epoch': int(epoch), 'values': [float(x) for x in rmse]})
        history['test_metrics'][var]['mape'].append({'epoch': int(epoch), 'values': [float(x) for x in mape]})
        history['test_metrics'][var]['acc'].append({'epoch': int(epoch), 'values': [float(x) for x in acc]})


def _plot_loss_history(history, log_dir):
    save_path = os.path.join(log_dir, 'loss_curve.png')
    draw_loss_line(history['train_loss'], history['val_loss'], save_path)


def _plot_fusion_history(history, log_dir):
    for split in ['train', 'val', 'test', 'final']:
        plt.figure(figsize=(12, 4))
        for i, tgt in enumerate(MODALITY_ORDER, start=1):
            ax = plt.subplot(1, 3, i)
            for src in MODALITY_ORDER:
                x = history['fusion'][split][tgt][src]['x']
                y = history['fusion'][split][tgt][src]['y']
                if len(x) > 0:
                    ax.plot(x, y, label=src)
            ax.set_title(f'{split}:{tgt}')
            ax.set_xlabel('epoch')
            ax.set_ylabel('weight')
            ax.set_ylim(0.0, 1.0)
            ax.grid(alpha=0.25)
            if i == 1 and len(ax.lines) > 0:
                ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'fusion_{split}.png'))
        plt.close()


def _plot_test_metric_history(history, log_dir):
    for var, metric_map in history['test_metrics'].items():
        component_names = COMPONENT_NAMES.get(var, None)
        for metric_name, records in metric_map.items():
            if len(records) == 0:
                continue
            # records: [{epoch, values:[C]}]
            epochs = [r['epoch'] for r in records]
            values = [r['values'] for r in records]
            c = len(values[0]) if len(values) > 0 else 0

            plt.figure(figsize=(8, 4))
            for ci in range(c):
                ys = [v[ci] for v in values]
                label = component_names[ci] if component_names is not None and ci < len(component_names) else f'c{ci}'
                # keep original unit convention: mape shown as ratio in history plot for consistency with logs
                if metric_name == 'mape':
                    ys = [yy * 100.0 for yy in ys]
                plt.plot(epochs, ys, marker='o', label=label)

            y_label = metric_name.upper() + (' (%)' if metric_name == 'mape' else '')
            plt.xlabel('epoch')
            plt.ylabel(y_label)
            plt.title(f'Test {metric_name.upper()} - {var}')
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f'test_{var}_{metric_name}.png'))
            plt.close()


def _save_history_files(history, log_dir):
    # JSON snapshot
    with open(os.path.join(log_dir, 'training_history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Flatten test metrics to CSV
    csv_path = os.path.join(log_dir, 'test_metrics_history.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['var', 'metric', 'epoch', 'component', 'value'])
        for var, metric_map in history['test_metrics'].items():
            component_names = COMPONENT_NAMES.get(var, None)
            for metric_name, records in metric_map.items():
                for r in records:
                    ep = r['epoch']
                    for i, v in enumerate(r['values']):
                        comp = component_names[i] if component_names is not None and i < len(component_names) else f'c{i}'
                        vv = float(v) * (100.0 if metric_name == 'mape' else 1.0)
                        writer.writerow([var, metric_name, ep, comp, vv])

    # Fusion history CSV
    csv_path2 = os.path.join(log_dir, 'fusion_history.csv')
    with open(csv_path2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'epoch', 'target', 'source', 'weight'])
        for split in ['train', 'val', 'test', 'final']:
            for tgt in MODALITY_ORDER:
                for src in MODALITY_ORDER:
                    xs = history['fusion'][split][tgt][src]['x']
                    ys = history['fusion'][split][tgt][src]['y']
                    for ep, w in zip(xs, ys):
                        writer.writerow([split, ep, tgt, src, float(w)])


def _save_and_plot_history(history, log_dir):
    _save_history_files(history, log_dir)
    _plot_loss_history(history, log_dir)
    _plot_fusion_history(history, log_dir)
    _plot_test_metric_history(history, log_dir)


def _tb_log_fusion(writer, split, epoch, fusion_stats):
    if writer is None or fusion_stats is None:
        return
    for tgt in MODALITY_ORDER:
        for src in MODALITY_ORDER:
            v = fusion_stats.get(tgt, {}).get(src, None)
            if v is not None:
                writer.add_scalar(f'fusion/{split}/{tgt}_from_{src}', float(v), int(epoch))


def _tb_log_test_metrics(writer, epoch, results):
    if writer is None:
        return
    for var, metrics in results.items():
        mae, rmse, mape, acc = metrics
        names = COMPONENT_NAMES.get(var, [f'{var}_{i}' for i in range(len(mae))])
        for i, n in enumerate(names):
            if i < len(mae):
                writer.add_scalar(f'test/{var}/MAE/{n}', float(mae[i]), int(epoch))
            if i < len(rmse):
                writer.add_scalar(f'test/{var}/RMSE/{n}', float(rmse[i]), int(epoch))
            if i < len(mape):
                writer.add_scalar(f'test/{var}/MAPE_percent/{n}', float(mape[i]) * 100.0, int(epoch))
            if i < len(acc):
                writer.add_scalar(f'test/{var}/ACC/{n}', float(acc[i]), int(epoch))

class OceanScaler:
    def __init__(self, mean_path, std_path, device):
        self.mean = torch.tensor(np.load(mean_path).flatten(), dtype=torch.float32).to(device)
        self.std = torch.tensor(np.load(std_path).flatten(), dtype=torch.float32).to(device)
        self.var_indices = {
            'flow': [0, 1],
            'wave': [2, 3, 4, 5], # 现在 wave 有 4 个分量：波高、波向sin、波向cos、周期
            'wind': [6, 7]        # 风速被挤到了最后两位 6, 7
        }

    def inverse_transform(self, data, var_name):
        idx = self.var_indices[var_name]
        m = self.mean[idx].view(1, 1, 1, -1)
        s = self.std[idx].view(1, 1, 1, -1)
        return data * s + m

def TrainEpoch(args, loader, model, optim, loss_fn, prompt_prefix, scaler, need_step: bool, amp_scaler=None):
    model.train() if need_step else model.eval()
    loss_item, count = 0, 0
    chosen_vars = args.predict_vars.split(',') 
    fusion_stats = _init_fusion_stats()

    for input, target, timestamp, cond_mask, ob_mask in loader:
        input, target, timestamp = input.to(device), target.to(device), timestamp.to(device)
        cond_mask, ob_mask = cond_mask.to(device), ob_mask.to(device)
        B, T, N, F = input.shape

        input_data = torch.where(cond_mask == 0, 0, input).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
        # Use AMP autocast in forward if requested
        if getattr(args, 'fp16', False):
            with torch.amp.autocast(device_type='cuda'):
                predict_dict, other_loss, aux_info = _unpack_model_output(model(input_data, timestamp, prompt_prefix, cond_mask))
        else:
            predict_dict, other_loss, aux_info = _unpack_model_output(model(input_data, timestamp, prompt_prefix, cond_mask))
        fusion_stats = _accumulate_fusion_stats(fusion_stats, aux_info.get('fusion_weights', {}))

        total_loss = 0
        valid_channels = 0

        # 💡 核心升级：通道级统一标准化 Loss
        # 不再使用任何手动权重，而是拆解到最细粒度的各个物理分量。
        # 由于它们都已被 Z-score 归一化，分布完全一致，因此各通道的 Loss 求平均就是绝对的均等。
        for var in chosen_vars:
            idx = scaler.var_indices[var]
            pred = predict_dict[var].view(B, N, -1, len(idx)).permute(0, 2, 1, 3).contiguous()
            targ_sub = target[:, -args.predict_len:, :, idx]
            mask_sub = ob_mask[:, -args.predict_len:, :, idx].bool()

            for i in range(len(idx)):
                p_ch = pred[..., i]
                t_ch = targ_sub[..., i]
                m_ch = mask_sub[..., i]
                
                if m_ch.sum() > 0:
                    total_loss += loss_fn(p_ch[m_ch], t_ch[m_ch])
                    valid_channels += 1

        # 将所有有效通道的 Loss 平均，做到各要素统一标准化相加
        if valid_channels > 0:
            total_loss = total_loss / valid_channels

        loss_item += total_loss.item(); count += 1
        
        if need_step:
            optim.zero_grad()
            L = total_loss
            for l in other_loss: L += l
            if getattr(args, 'fp16', False):
                amp_scaler.scale(L).backward()
                amp_scaler.step(optim)
                amp_scaler.update()
            else:
                L.backward()
                optim.step()

    return (loss_item / count if count else 0), _finalize_fusion_stats(fusion_stats)

def TestEpoch(args, loader, model, prompt_prefix, scaler, save=False, LOG_DIR=None, amp_scaler=None):
    model.eval()
    chosen_vars = args.predict_vars.split(',')
    storage = {v: {"preds": [], "targs": [], "masks": []} for v in chosen_vars}
    fusion_stats = _init_fusion_stats()

    with torch.no_grad():
        for input, target, timestamp, cond_mask, ob_mask in loader:
            input, target, timestamp = input.to(device), target.to(device), timestamp.to(device)
            cond_mask, ob_mask = cond_mask.to(device), ob_mask.to(device)
            B, T, N, F = input.shape

            input_proc = torch.where(cond_mask == 0, 0, input).permute(0, 2, 1, 3).contiguous().view(B, N, -1)
            if getattr(args, 'fp16', False):
                with torch.amp.autocast(device_type='cuda'):
                    predict_dict, _, aux_info = _unpack_model_output(model(input_proc, timestamp, prompt_prefix, cond_mask))
            else:
                predict_dict, _, aux_info = _unpack_model_output(model(input_proc, timestamp, prompt_prefix, cond_mask))
            fusion_stats = _accumulate_fusion_stats(fusion_stats, aux_info.get('fusion_weights', {}))

            for var in chosen_vars:
                idx = scaler.var_indices[var]
                pred = predict_dict[var].view(B, N, -1, len(idx)).permute(0, 2, 1, 3).contiguous()
                
                # 独立物理反归一化，评价真实量级误差
                pred_phys = scaler.inverse_transform(pred, var)
                targ_phys = scaler.inverse_transform(target[:, -args.predict_len:, :, idx], var)
                mask_sub = ob_mask[:, -args.predict_len:, :, idx].bool()

                storage[var]["preds"].append(pred_phys.detach())
                storage[var]["targs"].append(targ_phys.detach())
                storage[var]["masks"].append(mask_sub.detach())

        results = {}
        for var in chosen_vars:
            all_preds = torch.cat(storage[var]["preds"], 0)
            all_targs = torch.cat(storage[var]["targs"], 0)
            all_masks = torch.cat(storage[var]["masks"], 0)
            
            mae, rmse, mape, acc, _, _ = cal_metrics(all_preds, all_targs, all_masks)
            results[var] = (mae, rmse, mape, acc)
            
            if save and LOG_DIR:
                save_path = os.path.join(LOG_DIR, f'test_{var}.npz')
                np.savez(save_path, targets=all_targs.cpu().numpy(), predicts=all_preds.cpu().numpy(), mask=all_masks.cpu().numpy())
    return results, _finalize_fusion_stats(fusion_stats)

def Train(args, mylogger, model, prompt_prefix, scaler, train_loader, val_loader, test_loader, LOG_DIR):
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=5)
    loss_fn = torch.nn.L1Loss()
    
    best_loss = 1e9
    best_model = None
    patience_count = 0
<<<<<<< HEAD
    chosen_vars = args.predict_vars.split(',')
    history = _init_training_history(chosen_vars)
=======
>>>>>>> c8025504773d22d6913747305016a1ae5204d64c
    amp_scaler = None
    if getattr(args, 'fp16', False):
        # Use positional 'cuda' for compatibility with current torch version
        amp_scaler = torch.amp.GradScaler('cuda')
<<<<<<< HEAD

    tb_writer = None
    if getattr(args, 'tensorboard', False):
        if SummaryWriter is None:
            mylogger.warning('TensorBoard is requested but SummaryWriter is unavailable. Please install tensorboard package.')
        else:
            tb_dir = os.path.join(LOG_DIR, args.tb_subdir)
            check_dir(tb_dir, mkdir=True)
            tb_writer = SummaryWriter(log_dir=tb_dir)
            mylogger.info(f'[TensorBoard] logging to: {tb_dir}')

    for epoch in range(args.epoch):
        train_loss, train_fusion = TrainEpoch(args, train_loader, model, optim, loss_fn, prompt_prefix, scaler, need_step=True, amp_scaler=amp_scaler)
        history['train_loss']['x'].append(epoch)
        history['train_loss']['y'].append(float(train_loss))
        _record_fusion_history(history, 'train', epoch, train_fusion)
        if tb_writer is not None:
            tb_writer.add_scalar('loss/train', float(train_loss), epoch)
            _tb_log_fusion(tb_writer, 'train', epoch, train_fusion)
=======

    for epoch in range(args.epoch):
        train_loss, train_fusion = TrainEpoch(args, train_loader, model, optim, loss_fn, prompt_prefix, scaler, need_step=True, amp_scaler=amp_scaler)
>>>>>>> c8025504773d22d6913747305016a1ae5204d64c
        mylogger.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}")
        mylogger.info(f"[Fusion][Train] Epoch {epoch} {_format_fusion_stats(train_fusion)}")

        if epoch % args.val_epoch == 0:
            with torch.no_grad():
                val_loss, val_fusion = TrainEpoch(args, val_loader, model, optim, loss_fn, prompt_prefix, scaler, need_step=False, amp_scaler=amp_scaler)
<<<<<<< HEAD
            history['val_loss']['x'].append(epoch)
            history['val_loss']['y'].append(float(val_loss))
            _record_fusion_history(history, 'val', epoch, val_fusion)
            if tb_writer is not None:
                tb_writer.add_scalar('loss/val', float(val_loss), epoch)
                _tb_log_fusion(tb_writer, 'val', epoch, val_fusion)
=======
>>>>>>> c8025504773d22d6913747305016a1ae5204d64c
            mylogger.info(f"[Validation] Epoch {epoch} Val Loss: {val_loss:.4f}")
            mylogger.info(f"[Fusion][Val] Epoch {epoch} {_format_fusion_stats(val_fusion)}")
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model.grad_state_dict())
                patience_count = 0
            else:
                patience_count += 1

        if epoch % args.test_epoch == 0:
            res, test_fusion = TestEpoch(args, test_loader, model, prompt_prefix, scaler, amp_scaler=amp_scaler)
<<<<<<< HEAD
            _record_fusion_history(history, 'test', epoch, test_fusion)
            _record_test_metrics_history(history, epoch, res)
            if tb_writer is not None:
                _tb_log_fusion(tb_writer, 'test', epoch, test_fusion)
                _tb_log_test_metrics(tb_writer, epoch, res)
=======
>>>>>>> c8025504773d22d6913747305016a1ae5204d64c
            mylogger.info(f"[Fusion][Test] Epoch {epoch} {_format_fusion_stats(test_fusion)}")
            for var, metrics in res.items():
                mae, rmse, mape, acc = metrics
                mylogger.info(
                    f"   -> [Test][{var}] Epoch {epoch} "
                    f"{_format_named_metric(mae, var, 'MAE')} | "
                    f"{_format_named_metric(rmse, var, 'RMSE')} | "
                    f"{_format_named_metric(mape, var, 'MAPE')} | "
                    f"{_format_named_metric(acc, var, 'ACC')}"
                )

        if patience_count >= args.patience:
            mylogger.info('early stop')
            break
        
    if best_model:
        model.load_state_dict(best_model, strict=False)
        final_results, final_fusion = TestEpoch(args, test_loader, model, prompt_prefix, scaler, save=args.save_result, LOG_DIR=LOG_DIR, amp_scaler=amp_scaler)
<<<<<<< HEAD
        _record_fusion_history(history, 'final', -1, final_fusion)
        if tb_writer is not None:
            _tb_log_fusion(tb_writer, 'final', 0, final_fusion)
            _tb_log_test_metrics(tb_writer, 0, final_results)
=======
>>>>>>> c8025504773d22d6913747305016a1ae5204d64c
        mylogger.info(f"[Fusion][Final] {_format_fusion_stats(final_fusion)}")
        for var, metrics in final_results.items():
            mae, rmse, mape, acc = metrics
            mylogger.info(
                f"[Final Best][{var}] "
                f"{_format_named_metric(mae, var, 'MAE')} | "
                f"{_format_named_metric(rmse, var, 'RMSE')} | "
                f"{_format_named_metric(mape, var, 'MAPE')} | "
                f"{_format_named_metric(acc, var, 'ACC')}"
            )

    # Persist and visualize history artifacts for post-analysis
    _save_and_plot_history(history, LOG_DIR)

    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

if __name__ == '__main__':
    args = InitArgs()

    # Map --model string to the correct BaseModel class.
    model_name = args.model.lower() if isinstance(args.model, str) else ''
    if model_name == 'transformer':
        ModelClass = Transformer
    elif model_name == 'phi2':
        ModelClass = Phi2
    elif model_name == 'gpt2':
        ModelClass = GPT2
    elif model_name.startswith('qwen'):
        # allow 'qwen', 'qwen3' etc.
        ModelClass = QWEN
    elif model_name.startswith('llama') or model_name.startswith('llama3'):
        ModelClass = LLAMA3
    else:
        # If an unknown model string is provided, raise an informative error rather than silently
        # falling back to QWEN (which caused confusion).
        raise ValueError(f"Unknown --model '{args.model}'. Supported: phi2, gpt2, transformer, qwen, llama3")

    print(f"Selected base model: {model_name}")
    basemodel = ModelClass(args.causal, args.lora, args.ln_grad, args.llm_layers)
    
    ocean_data_dir = './data/stdplm_input_025'
    # 💡 核心修复：强制设置 num_workers=0，禁止 Windows 多进程复制撑爆内存
    dataloaders = get_ocean_dataloaders(ocean_data_dir, batch_size=args.batch_size, num_workers=0)
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']
    
    scaler = OceanScaler(f'{ocean_data_dir}/norm_mean.npy', f'{ocean_data_dir}/norm_std.npy', device)
    node_embeddings = load_ocean_laplacian_embeddings(ocean_data_dir, K=args.trunc_k).to(device)

    safe_time_str = get_time_str().replace(':', '-')
    LOG_DIR = os.path.join(args.log_root, f'{safe_time_str}_{args.desc}_{random_str()}')
    check_dir(LOG_DIR, mkdir=True)
    mylogger = getlogger(os.path.join(LOG_DIR, 'experiments.log'))

    output_len = args.predict_len
    if args.task == 'all': output_len += args.sample_len
    elif args.task == 'imputation': output_len = args.sample_len

    prompt_ids = None
    if args.prompt_prefix is not None:
        tokenizer = basemodel.gettokenizer()
        prompt_ids = tokenizer(args.prompt_prefix, return_tensors="pt", return_attention_mask=False)
        prompt_ids = prompt_ids['input_ids'].to(device).view(-1, 1)

    model = STALLM_MIMO(basemodel=basemodel, sample_len=args.sample_len, output_len=output_len,
                        input_dim=args.input_dim, output_dim=args.output_dim,
                        node_emb_dim=args.node_emb_dim, sag_dim=args.sag_dim, sag_tokens=args.sag_tokens,
                        node_embeddings=node_embeddings, use_node_embedding=args.node_embedding,
                        use_timetoken=args.time_token, use_sandglassAttn=args.sandglassAttn,
                        dropout=args.dropout, trunc_k=args.trunc_k, t_dim=args.t_dim,
                        fusion_mode=args.fusion_mode,
                        use_revin=args.revin, revin_affine=args.revin_affine).to(device)

    total_params, total_trainable_params = model.params_num()
    mylogger.info(f'Total Params: {total_params} | Trainable: {total_trainable_params}')

    Train(args, mylogger, model, prompt_ids, scaler, train_loader, val_loader, test_loader, LOG_DIR)
    model.save(os.path.join(LOG_DIR, f'{safe_time_str}_{args.desc}.pth'))