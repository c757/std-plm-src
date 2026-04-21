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
    def __init__(self, data_path, indices_path, split='train', sample_len=9, predict_len=3, input_layout='node'):
        data_dir = os.path.dirname(data_path)
        # 【修复2】：必须用 'r' (只读)，保护你的 24GB 内存不被多进程吃光！
        self.data = np.load(data_path, mmap_mode='r')
        self.is_grid_layout = (self.data.ndim == 4)  # (T, H, W, C)
        self.input_layout = (input_layout or 'node').lower()
        self.indices = np.load(indices_path)[f'{split}_idx']
        self.sample_len = sample_len
        self.predict_len = predict_len
        self.window = sample_len + predict_len
        
        # 【修复3】：让搬运工认识真实的海洋地图和真实的时间
        mask_2d_path = f'{data_dir}/ocean_mask_2d.npy'
        mask_1d_path = f'{data_dir}/ocean_mask.npy'
        self.ocean_mask_2d = None
        self.ocean_mask_1d = None
        if os.path.exists(mask_2d_path):
            m2d = np.load(mask_2d_path).astype(np.float32)
            self.ocean_mask_2d = torch.tensor(m2d, dtype=torch.float32)
            self.ocean_mask_1d = self.ocean_mask_2d.reshape(-1)
        elif os.path.exists(mask_1d_path):
            m1d = np.load(mask_1d_path).astype(np.float32).reshape(-1)
            self.ocean_mask_1d = torch.tensor(m1d, dtype=torch.float32)
        else:
            raise FileNotFoundError(f'Cannot find ocean mask file: {mask_2d_path} or {mask_1d_path}')

        # Runtime mask shape policy:
        # - grid path uses 2D mask when available
        # - node path always uses flattened mask
        if self.input_layout == 'grid' and self.is_grid_layout and self.ocean_mask_2d is not None:
            self.ocean_mask = self.ocean_mask_2d
        else:
            self.ocean_mask = self.ocean_mask_1d
        self.timestamps = pd.to_datetime(np.load(f'{data_dir}/timestamps.npy'))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_t = self.indices[idx]
        end_t = start_t + self.window
        
        window_data = self.data[start_t:end_t]
        if self.is_grid_layout and self.input_layout == 'grid':
            # Commit-1: output true 5D tensors for model bridge path.
            # (T, H, W, C) -> (T, C, H, W)
            x = torch.tensor(window_data[:self.sample_len].copy(), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
            y = torch.tensor(window_data[self.sample_len:].copy(), dtype=torch.float32).permute(0, 3, 1, 2).contiguous()
        else:
            if self.is_grid_layout:
                # node compatibility path: flatten spatial grid into N.
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
        if self.is_grid_layout and self.input_layout == 'grid' and self.ocean_mask_2d is not None:
            # x: (sample_len, C, H, W), y: (predict_len, C, H, W)
            mask_view = self.ocean_mask_2d.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            cond_mask = torch.ones_like(x) * mask_view
            # Keep window-length ob_mask contract for existing training code.
            ob_mask = torch.ones((self.window, x.shape[1], x.shape[2], x.shape[3]), dtype=torch.float32) * mask_view
        else:
            cond_mask = torch.ones_like(x) * self.ocean_mask_1d.view(1, -1, 1)
            ob_mask = torch.ones((self.window, x.shape[1], x.shape[2]), dtype=torch.float32) * self.ocean_mask_1d.view(1, -1, 1)
        
        return x, y, timestamp, cond_mask, ob_mask

def get_ocean_dataloaders(data_dir, batch_size=16, num_workers=4, input_layout='node'):
    data_nodes_path = f'{data_dir}/data_nodes.npy'
    data_grid_path = f'{data_dir}/data_grid.npy'
    layout = (input_layout or 'node').lower()

    if layout == 'node':
        if os.path.exists(data_nodes_path):
            data_path = data_nodes_path
        elif os.path.exists(data_grid_path):
            # Commit-0 default prefers node, but fallback keeps training runnable.
            print('[DataLoader] input_layout=node but data_nodes.npy is missing; fallback to data_grid.npy')
            data_path = data_grid_path
        else:
            raise FileNotFoundError(f'Cannot find data file: {data_nodes_path} or {data_grid_path}')
    elif layout == 'grid':
        if os.path.exists(data_grid_path):
            data_path = data_grid_path
        elif os.path.exists(data_nodes_path):
            print('[DataLoader] input_layout=grid but data_grid.npy is missing; fallback to data_nodes.npy')
            data_path = data_nodes_path
        else:
            raise FileNotFoundError(f'Cannot find data file: {data_grid_path} or {data_nodes_path}')
    else:
        raise ValueError(f'Unsupported input_layout={input_layout}, expected one of: node, grid')

    indices_path = f'{data_dir}/indices.npz'
    import json
    with open(f'{data_dir}/meta.json', 'r') as f:
        meta = json.load(f)

    print(f'[DataLoader] input_layout={layout}, using file: {os.path.basename(data_path)}')
    
    datasets = {}
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = OceanDataset(
            data_path,
            indices_path,
            split=split,
            sample_len=meta['sample_len'],
            predict_len=meta['predict_len'],
            input_layout=layout,
        )
        dataloaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers, pin_memory=True)
    return dataloaders
