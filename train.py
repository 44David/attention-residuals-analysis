import os
import json
from types import SimpleNamespace
import torch
import numpy as np

# open config file
with open("config.json") as f:
    config_dict = json.load(f)
    
config = SimpleNamespace(**config_dict)

out_dir="out"

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

m_process = True
ddp_world_size = 1 

tokens_per_iter = config.gradient_accumulation_steps * ddp_world_size * config.batch_size * config.block_size
print(f"tokens per iter: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join('data', config.dataset)
def get_batch(split):
    if split=='train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'eval.bin'), dtype=np.uint16, mode='r')
    
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])

    x, y = x.pin_memory().to('cuda', non_blocking=True), y.pin_memory.to('cuda', non_blocking=True)
    
    return x, y


iter_num = 0
best_val_loss = 1e9
