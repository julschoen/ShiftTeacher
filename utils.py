import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import functools
from torch.nn import Parameter as P

def snconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))

def snconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return SpectralNorm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))

def snlinear(in_features, out_features):
    return SpectralNorm(nn.Linear(in_features=in_features, out_features=out_features))

class Attention2D(nn.Module):
  def __init__(self, in_dim):
      super(Attention2D, self).__init__()
      self.chanel_in = in_dim

      self.query_conv = snconv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False)
      self.key_conv = snconv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False)
      self.value_conv = snconv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0, bias=False)

      self.gamma = P(torch.zeros(1))

      self.softmax = nn.Softmax(dim=-1)
  def forward(self, x):
      m_batchsize, C, height, width = x.size()
      proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
      proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
      energy = torch.bmm(proj_query, proj_key)
      attention = self.softmax(energy)
      proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

      out = torch.bmm(proj_value, attention.permute(0, 2, 1))
      out = out.view(m_batchsize, C, height, width)

      out = self.gamma*out + x
      #out = F.avg_pool2d(out, out.size()[2:4])

      return out

class Attention3D(nn.Module):
  def __init__(self, ch):
    super(Attention3D, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.theta = snconv3d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = snconv3d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = snconv3d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = snconv3d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool3d(self.phi(x), [2,2,2], stride=2)
    g = F.max_pool3d(self.g(x), [2,2,2], stride=2)    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3] * x.shape[4])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] * x.shape[4] // 8)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] * x.shape[4] // 8)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.permute(0,2,1), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.permute(0,2,1)).view(-1, self.ch // 2, x.shape[2], x.shape[3], x.shape[4]))
    return self.gamma * o + x

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, conv, wide=True, preactivation=True,
               downsample=None, channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio

    self.preactivation = preactivation
    self.activation = nn.ReLU(inplace=True)
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = conv(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = conv(self.hidden_channels, self.hidden_channels)
    self.conv3 = conv(self.hidden_channels, self.hidden_channels)
    self.conv4 = conv(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = conv(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)