import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import os

def load_ocean_laplacian_embeddings(data_dir, K=64):
    print("正在计算稀疏拉普拉斯拓扑嵌入...")
    import json
    with open(f'{data_dir}/meta.json', 'r') as f:
        n_nodes = json.load(f)['n_nodes']
        
    edges = pd.read_csv(f'{data_dir}/ocean_adj.csv')
    A = sp.coo_matrix((edges['cost'], (edges['from'], edges['to'])), shape=(n_nodes, n_nodes), dtype=np.float32)
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)
    A = A + sp.eye(n_nodes)
    
    degree = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(n_nodes) - D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    
    # 【修复1】：使用 sigma=0 的平移反转法，又快又准
    eigenvalues, eigenvectors = eigsh(L, k=K, which='LM', sigma=0)
    return torch.tensor(eigenvectors, dtype=torch.float32)
    
class OceanDataset(Dataset):
    def __init__(self, data_path, indices_path, split='train', sample_len=9, predict_len=3):
        data_dir = os.path.dirname(data_path)
        # 【修复2】：必须用 'r' (只读)，保护你的 24GB 内存不被多进程吃光！
        self.data = np.load(data_path, mmap_mode='r')
        self.is_grid_layout = (self.data.ndim == 4)  # (T, H, W, C)
        self.indices = np.load(indices_path)[f'{split}_idx']
        self.sample_len = sample_len
        self.predict_len = predict_len
        self.window = sample_len + predict_len
        
        # 【修复3】：让搬运工认识真实的海洋地图和真实的时间
        mask_2d_path = f'{data_dir}/ocean_mask_2d.npy'
        mask_1d_path = f'{data_dir}/ocean_mask.npy'
        if os.path.exists(mask_2d_path):
            mask_np = np.load(mask_2d_path).astype(np.float32).reshape(-1)
        elif os.path.exists(mask_1d_path):
            mask_np = np.load(mask_1d_path).astype(np.float32).reshape(-1)
        else:
            raise FileNotFoundError(f'Cannot find ocean mask file: {mask_2d_path} or {mask_1d_path}')

        self.ocean_mask = torch.tensor(mask_np, dtype=torch.float32)
        self.timestamps = pd.to_datetime(np.load(f'{data_dir}/timestamps.npy'))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        end_t = start_t + self.window
        
        window_data = self.data[start_t:end_t]
        if self.is_grid_layout:
            # Convert (T, H, W, C) -> (T, N, C) to keep current training/model contract unchanged.
            tw, h, w, c = window_data.shape
            window_data = window_data.reshape(tw, h * w, c)

        x = torch.tensor(window_data[:self.sample_len].copy(), dtype=torch.float32)
        y = torch.tensor(window_data[self.sample_len:].copy(), dtype=torch.float32)
        
        # 【修复4】：传入真实的时间 (星期、小时、分钟)
        t_slice = self.timestamps[start_t:end_t]
        timestamp = torch.zeros((self.window, 5), dtype=torch.float32)
        timestamp[:, 2] = torch.tensor(t_slice.dayofweek.values)
        timestamp[:, 3] = torch.tensor(t_slice.hour.values)
        timestamp[:, 4] = torch.tensor(t_slice.minute.values)
        
        # 【修复5】：给陆地打上 0 的标签，模型算误差时会自动忽略陆地
        cond_mask = torch.ones_like(x) * self.ocean_mask.view(1, -1, 1)
        ob_mask = torch.ones((self.window, x.shape[1], x.shape[2])) * self.ocean_mask.view(1, -1, 1)
        
        return x, y, timestamp, cond_mask, ob_mask

def get_ocean_dataloaders(data_dir, batch_size=16, num_workers=4):
    data_nodes_path = f'{data_dir}/data_nodes.npy'
    data_grid_path = f'{data_dir}/data_grid.npy'
    if os.path.exists(data_grid_path):
        data_path = data_grid_path
    elif os.path.exists(data_nodes_path):
        data_path = data_nodes_path
    else:
        raise FileNotFoundError(f'Cannot find data file: {data_grid_path} or {data_nodes_path}')

    indices_path = f'{data_dir}/indices.npz'
    import json
    with open(f'{data_dir}/meta.json', 'r') as f:
        meta = json.load(f)
    
    datasets = {}
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = OceanDataset(data_path, indices_path, split=split, sample_len=meta['sample_len'], predict_len=meta['predict_len'])
        dataloaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers, pin_memory=True)
    return dataloaders
