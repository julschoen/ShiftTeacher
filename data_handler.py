import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.len = self.files.shape[0]

  def __getitem__(self, index):
      x = self.files[index]
      ind = np.sort(np.random.choice(x.shape[0], 2, replace=False))
      while ind[0]+1 == ind[1]:
        ind = np.sort(np.random.choice(x.shape[0], 2, replace=False))
      ind = np.sort(np.append(ind, (ind[0]+ind[1]//2)))
      xs = x[ind]
      y = (ind[2]-ind[0])/(x.shape[0]-ind[0])
      xs = np.clip(xs, -1, 1)
      return torch.from_numpy(xs).float().squeeze(), torch.Tensor([y])

  def __len__(self):
      return self.len
