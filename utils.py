import torch
import numpy as np
def load_data(path):
    return torch.from_numpy(np.loadtxt(path, delimiter=",", skiprows=1))