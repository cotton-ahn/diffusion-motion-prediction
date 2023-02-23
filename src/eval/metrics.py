# code borrowed from DLow [https://github.com/Khrylx/DLow]

import numpy as np
from scipy.spatial.distance import pdist

"""metrics"""

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_min_de(pred, gt, *args):
    diff = pred - gt 
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()

def compute_avg_de(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return np.mean(dist)


def compute_std_de(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return np.std(dist)


def compute_min_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()

def compute_avg_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.mean()

def compute_std_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.std()
