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
      shift = (ind[2]-ind[0])/140
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
      shift = (ind1[2]-ind1[0])  /140
      xs = np.clip(xs, -1, 1)
      same = 0
    return torch.from_numpy(xs).float().squeeze(), torch.Tensor([shift]).float(), torch.Tensor([same]).float()

  def __len__(self):
      return self.len

class DATA3D(Dataset):
  def __init__(self, path): 
    self.files = np.load(path)['x']
    self.path = path[:-len(path.split('/')[-1])]
    self.len = len(self.files)

  def __getitem__(self, index):
      if np.random.rand() < 0.51:
        #pat = [os.path.join(self.path, self.files[index], f) for f in os.listdir(os.path.join(self.path, self.files[index])) if f.endswith('npz')][0]
        pat = os.path.join(self.path, self.files[index])
        x = np.load(pat)['x']
        if x.shape[0] < 6:
          shape = x.shape[0]
        elif x.shape[0] == 9:
          shape = 5
        else:
          shape = 6
        ind = np.sort(np.random.choice(shape, 2, replace=False))
        while ind[0]+1 == ind[1]:
          ind = np.sort(np.random.choice(shape, 2, replace=False))
        ind = np.sort(np.append(ind, np.mean(ind).astype(int)))
        xs = x[ind]
        shift = (ind[2]-ind[0])/10
        same = 1
      else:
        pat1 = os.path.join(self.path, self.files[index])
        r_pat = np.random.choice(self.len, 1)[0]
        pat2 = os.path.join(self.path, self.files[r_pat])
        x1 = np.load(pat1)['x']
        x2 = np.load(pat2)['x']
        num_others = np.random.randint(1,3)
        if x1.shape[0] < 6:
          shape1 = x1.shape[0]
        elif x1.shape[0] == 9:
          shape1 = 5
        else:
          shape1 = 6
        if x2.shape[0] < 6:
          shape2 = x2.shape[0]
        elif x2.shape[0] == 9:
          shape2 = 5
        else:
          shape2 = 6
        ind1 = np.sort(np.random.choice(shape1, 2, replace=False))
        ind2 = np.sort(np.random.choice(shape2, num_others, replace=False))
        while ind1[0]+1 == ind1[1]:
          ind1 = np.sort(np.random.choice(shape1, 2, replace=False))
        ind1 = np.sort(np.append(ind1, np.mean(ind1).astype(int)))
        xs = x1[ind1]
        xs[np.random.choice(xs.shape[0], num_others, replace=False)] = x2[ind2]
        shift = (ind1[2]-ind1[0])/10
        same = 0
      xs_ = np.empty((3,64,128,128))
      for i, x in enumerate(xs):
        xs_[i] = np.flip(x.reshape(128,128,64).T,axis=0)
      xs = np.clip(xs_, -1,1)
      return torch.from_numpy(xs).float().squeeze(), torch.Tensor([shift]).float(), torch.Tensor([same]).float()

  def __len__(self):
      return self.len
