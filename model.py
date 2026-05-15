import math
import torch 
import torch.nn as nn
from torch.nn import functional as F
import json



class DecoderBlock(nn.Module):
    def __init__(self, config):
        self.d_model = config.d_model
        
        self.ln1 = F.layer_norm(self.d_model)
        self.ln2 = F.layer_norm(self.d_model)
        
        self.attention = MultiHeadAttention(d_model=self.d_model)
        self.network = MLP(config)
        
        
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.network(self.ln2(x))
        
        return x
        
    
class Model(nn.Module):
    def __init__(self, config):
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.dropout = config.dropout
        self.n_layer = config.n_layer
        
        self.gpt = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.d_model),
            wpe = nn.Embedding(self.block_size, self.d_model),
            dropout = nn.Dropout(self.dropout), 
            hidden = nn.ModuleList([DecoderBlock(config) for _ in range(self.n_layer)]),
            ln_final = F.layer_norm(self.d_model)
        ))
        
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False) 
        
        # weight tying 
        self.gpt.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        # scaled init to residual projections
        for param_name, param in self.named_parameters():
            if param_name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2*self.n_layer))
                
                
    
        # TODO 
        # write method for getting num of params
        
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)
    
    
    def forward(self, idx, targets=None):
        device = idx.device
        
        b, t = idx.size()
        
        assert t <= self.block_size, f"Cannot forward seq. of length {t}, block size is {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.gpt.wte(idx)
        pos_emb = self.gpt.wpe(pos)
        
        x = self.gpt.drop(tok_emb + pos_emb)
        
        for block in self.gpt.hidden:
            x = DecoderBlock(x)
            
        x = self.gpt.ln_final(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None


        return logits, loss
