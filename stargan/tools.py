import torch
import numpy as np


def label2onehot(labels, dim, cuda):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    if cuda:
        out = out.to(torch.device('cuda'))
    return out