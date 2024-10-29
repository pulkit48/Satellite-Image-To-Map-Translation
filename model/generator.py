import torch
import torch.nn as nn
from torch.nn import functional as F

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim) if norm else None

    def forward(self, x):
        x = self.lrelu(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_dim)
        self.dropout = nn.Dropout2d(p=0.5, inplace=True) if dropout else None

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, in_dim, gating_dim, inter_dim):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_dim, inter_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_dim)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_dim, inter_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_dim)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_dim, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)

        # Attention Gates for each skip connection
        self.attention7 = AttentionBlock(512, 512, 256)
        self.attention6 = AttentionBlock(512, 512, 256)
        self.attention5 = AttentionBlock(512, 512, 256)
        self.attention4 = AttentionBlock(512, 512, 128)
        self.attention3 = AttentionBlock(256, 256, 64)
        self.attention2 = AttentionBlock(128, 128, 32)
        self.attention1 = AttentionBlock(64, 64, 16)

        # Decoder
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2 * 512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2 * 512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2 * 512, 512)
        self.decoder4 = DecoderBlock(2 * 512, 256)
        self.decoder3 = DecoderBlock(2 * 256, 128)
        self.decoder2 = DecoderBlock(2 * 128, 64)
        self.decoder1 = nn.ConvTranspose2d(2 * 64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        # Decoding with attention
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, self.attention7(e7, d8)], dim=1)
        
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, self.attention6(e6, d7)], dim=1)
        
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, self.attention5(e5, d6)], dim=1)
        
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, self.attention4(e4, d5)], dim=1)
        
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, self.attention3(e3, d4)], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, self.attention2(e2, d3)], dim=1)
        
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, self.attention1(e1, d2)], dim=1)
        
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
