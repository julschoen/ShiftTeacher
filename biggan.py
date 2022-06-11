import torch 
import torch.nn as nn
from torch.nn import init
from utils import Attention2D, Attention3D, DBlock, snconv2d, snconv3d, snlinear


class Discriminator(nn.Module):
  def __init__(self, params):
    super(Discriminator, self).__init__()
    self.p = params
    # Architecture
    if self.p.in_shape == 128:
      self.arch = {'in_channels' :  [item * self.p.filters for item in [1, 2, 4,  8, 16]],
                 'out_channels' : [item * self.p.filters for item in [2, 4, 8, 16, 16]],
                 'downsample' : [True] * 5 + [False],
                 'resolution' : [64, 32, 16, 8, 4, 4],
                 'attention' : {2**i: 2**i in [int(item) for item in '16'.split('_')]
                                for i in range(2,8)}}
    elif self.p.in_shape == 64:
      self.arch = {'in_channels' :  [item * self.p.filters for item in [1, 2, 4, 8]],
               'out_channels' : [item * self.p.filters for item in [2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in '16'.split('_')]
                              for i in range(2,7)}}
    else:
      raise NotImplementedError
    
    # Prepare model
    if self.p.threeD:
      self.input_conv = snconv3d(3, self.arch['in_channels'][0])
      conv = snconv3d
      att = Attention3D
      down = nn.AvgPool3d(2)
    else:
      self.input_conv = snconv2d(3, self.arch['in_channels'][0])
      conv = snconv2d
      att = Attention2D
      down = nn.AvgPool2d(2)

    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index] if d_index==0 else self.arch['out_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       conv=conv,
                       preactivation=True,
                       downsample=(down if self.arch['downsample'][index] and d_index==0 else None))
                       for d_index in range(1)]]
      if self.arch['attention'][self.arch['resolution'][index]]:
        self.blocks[-1] += [att(self.arch['out_channels'][index])]

    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    self.linear_shift = snlinear(self.arch['out_channels'][-1], 1)
    self.linear_dir = snlinear(self.arch['out_channels'][-1], 1)

    self.activation = nn.ReLU(inplace=True)
    self.act = nn.Sigmoid()
    self.init_weights()

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d)
          or isinstance(module, nn.Linear)):
        init.orthogonal_(module.weight)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x):
    # Run input conv
    h = self.input_conv(x)
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    if self.p.threeD:
      h = torch.sum(self.activation(h), [2, 3, 4])
    else:
      h = torch.sum(self.activation(h), [2, 3])
    return self.linear_shift(h), self.linear_dir(h)