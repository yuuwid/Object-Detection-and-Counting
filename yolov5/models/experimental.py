
import math

import numpy as np
import torch
import torch.nn as nn


# Weighted sum of 2 or more layers
# referensi: https://arxiv.org/abs/1911.09070
class Sum(nn.Module):
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()

        # apply weights boolean
        self.weight = weight

        # iter object
        self.iter = range(n - 1)

        if weight:
            # layer weights
            data = -torch.arange(1.0, n) / 2
            self.w = nn.Parameter(data, requires_grad=True)  


    def forward(self, x):
        # no weight
        y = x[0]  
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


# Mixed Depth-wise Conv
# referensi: https://arxiv.org/abs/1907.09595
class MixConv2d(nn.Module):
    
    # ch_in, ch_out, kernel, stride, ch_strategy
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):  
        super().__init__()
    
        n = len(k)  # number of convolutions
    
        # equal c_ per group
        if equal_ch:  
            # c2 indices
            i = torch.linspace(0, n - 1E-6, c2).floor()  
            
            # intermediate channels
            c_ = [(i == g).sum() for g in range(n)]  
        
        # equal weight.numel() per group
        else:  
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            
            # solve for equal weight indices, ax = b
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  

        self.m = nn.ModuleList([
                nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)
            ])
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))



# Ensemble of models
class Ensemble(nn.ModuleList):
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output



def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model
    
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        # load
        ckpt = torch.load(w, map_location='cpu')  
        
        # FP32 model
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  

        # Model compatibility updates
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])
        
        if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
            # convert to dict
            ckpt.names = dict(enumerate(ckpt.names))  

        # model in eval mode
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  
    
    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            # torch 1.7.0 compatibility
            m.inplace = inplace  
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)

        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            # torch 1.11.0 compatibility
            m.recompute_scale_factor = None  

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')

    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))

    torch_ix = torch.tensor([m.stride.max() for m in model])
    torch_ix = torch.argmax(torch_ix).int()

    # max stride
    model.stride = model[torch_ix].stride  
    
    assert1 = all(model[0].nc == m.nc for m in model)
    assert2 = f'Models have different class counts: {[m.nc for m in model]}'
    assert assert1, assert2
    
    return model
