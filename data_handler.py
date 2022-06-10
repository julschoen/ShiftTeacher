import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.len = self.files.shape[0]

  def __getitem__(self, index):
    if np.random.rand() < 0.51:
      x = self.files[index]
      ind = np.sort(np.random.choice(x.shape[0], 2, replace=False))
      while ind[0]+1 == ind[1]:
        ind = np.sort(np.random.choice(x.shape[0], 2, replace=False))
      ind = np.sort(np.append(ind, np.mean(ind).astype(int)))
      xs = x[ind]
      shift = (ind[2]-ind[0])/(x.shape[0]-ind[0])
      xs = np.clip(xs, -1, 1)
      same = 1
    else:
      x1 = self.files[index]
      x2 = self.files[np.random.choice(self.len, 1)[0]]
      num_others = np.random.randint(1,3)
      ind1 = np.sort(np.random.choice(x1.shape[0], 2, replace=False))
      ind2 = np.sort(np.random.choice(x2.shape[0], num_others, replace=False))
      while ind1[0]+1 == ind1[1]:
        ind1 = np.sort(np.random.choice(x1.shape[0], 2, replace=False))
      ind1 = np.sort(np.append(ind1, np.mean(ind1).astype(int)))
      xs = x1[ind1]
      xs[np.random.choice(xs.shape[0], num_others, replace=False)] = x2[ind2]
      shift = (ind1[2]-ind1[0])/(x1.shape[0]-ind1[0])
      xs = np.clip(xs, -1, 1)
      same = 0
    return torch.from_numpy(xs).float().squeeze(), torch.Tensor([shift]).float(), torch.Tensor([same]).float()

  def __len__(self):
      return self.len
