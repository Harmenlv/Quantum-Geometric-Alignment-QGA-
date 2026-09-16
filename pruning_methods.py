import torch
import torch.nn as nn

def magnitude_prune(model, ratio=0.9):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data.abs()
            threshold = torch.quantile(w.flatten(), ratio)
            mask = w >= threshold
            m.weight.data *= mask.float()
    return model

def random_prune(model, ratio=0.9):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            mask = torch.rand_like(m.weight.data) > ratio
            m.weight.data *= mask.float()
    return model