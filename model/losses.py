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