import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from types import SimpleNamespace
from model.llm import Transformer
from model.model import STALLM_MIMO
from main import TrainEpoch

# small config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build small basemodel
basemodel = Transformer(False, False, False, layers=1)

# small node embeddings: N=4, trunc_k=16
node_embeddings = torch.randn(4, 16)

# construct STALLM_MIMO with small sizes
model = STALLM_MIMO(basemodel=basemodel, sample_len=4, output_len=4,
                    input_dim=1, output_dim=1, node_emb_dim=8, sag_dim=16, sag_tokens=8,
                    node_embeddings=node_embeddings, use_node_embedding=True,
                    use_timetoken=True, use_sandglassAttn=False,
                    dropout=0, trunc_k=16, t_dim=8, fusion_mode='cosine',
                    use_revin=False, revin_affine=False).to(device)

# synthetic batch
B = 1
sample_len = 4
predict_len = 4
T = sample_len + predict_len
N = 4
F = 8

input_tensor = torch.randn(B, T, N, F, dtype=torch.float32).to(device)
target_tensor = torch.randn(B, T, N, F, dtype=torch.float32).to(device)
timestamp = torch.randint(0, 24*60*7, (B, T, 5), dtype=torch.long).to(device)
cond_mask = torch.ones_like(input_tensor)
ob_mask = torch.ones_like(input_tensor)

# args mock
args = SimpleNamespace()
args.predict_vars = 'flow,wave,wind'
args.predict_len = predict_len
args.fp16 = True

# optimizer, loss
optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
loss_fn = torch.nn.L1Loss()

# fake scaler for var indices
scaler = SimpleNamespace(var_indices={'flow':[0,1],'wave':[2,3,4,5],'wind':[6,7]})

# amp scaler
amp_scaler = torch.amp.GradScaler('cuda') if args.fp16 and torch.cuda.is_available() else None

loader = [(input_tensor, target_tensor, timestamp, cond_mask, ob_mask)]

print('Running TrainEpoch single-batch AMP test...')
train_loss, fusion_stats = TrainEpoch(args, loader, model, optim, loss_fn, None, scaler, need_step=True, amp_scaler=amp_scaler)
print('TrainEpoch returned loss:', train_loss)
print('Fusion stats:', fusion_stats)
