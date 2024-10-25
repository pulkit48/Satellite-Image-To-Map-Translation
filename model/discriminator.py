
import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):

  def __init__(self,in_dim,out_dims,kernel_size=4,stride=2,padding=1,norm=True):

    super().__init__()
    self.conv=nn.Conv2d(in_dim,out_dims,kernel_size=kernel_size,padding=padding,stride=stride)
    self.isn=None

    if norm:
      self.isn=nn.InstanceNorm2d(out_dims,affine=True)

    self.lrelu=nn.LeakyReLU(0.2,inplace=True)

  def forward(self,x):
    x=self.conv(x)

    if self.isn is not None:
      x=self.isn(x)

    x=self.lrelu(x)

    return torch.sigmoid(x)

class Discriminator(nn.Module):

  def __init__(self,):

    super().__init__()
    self.block1=BasicBlock(3,64,norm=False)
    self.block2=BasicBlock(64,128)
    self.block3=BasicBlock(128,256)
    self.block4=BasicBlock(256,512)
    self.block5=nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)

  def forward(self,x):

    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.block4(x)
    x=self.block5(x)

    return x

class ConditionalDiscriminator(nn.Module):

  def __init__(self,):

    super().__init__()
    self.block1=BasicBlock(6,64,norm=False)
    self.block2=BasicBlock(64,128)
    self.block3=BasicBlock(128,256)
    self.block4=BasicBlock(256,512)
    self.block5=nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)

  def forward(self,x,cond):

    x=torch.cat([x,cond],dim=1)
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=self.block4(x)
    x=self.block5(x)

    return x
import torch
import torch.nn as nn

class GeneratorLoss(nn.Module):

  def __init__(self,alpha=100):
    super().__init__()

    self.alpha=alpha
    self.bce=nn.BCEWithLogitsLoss()
    self.l1=nn.L1Loss()

  def forward(self,real,fake,fake_pred):

    fake_target=torch.ones_like(fake_pred)
    loss=self.bce(fake_pred,fake_target)+self.alpha*self.l1(real,fake)
    return loss


class DiscriminatorLoss(nn.Module):

  def __init__(self):

    super().__init__()

    self.bce=nn.BCEWithLogitsLoss()

  def forward(self,fake_pred,real_pred):
    fake_target=torch.zeros_like(fake_pred)
    real_target=torch.ones_like(real_pred)

    fake_loss=self.bce(fake_pred,fake_target)
    real_loss=self.bce(real_pred,real_target)

    loss=(fake_loss+real_loss)/2
    return loss
