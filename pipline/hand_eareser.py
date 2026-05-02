import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class ClockEraserV2(nn.Module):
    """
    4-level U-Net with skip connections.
    Input:  analog image with hands  (B, 3, H, W)
    Output: clean clock face         (B, 3, H, W)  values in [0, 1]
    """
    def __init__(self, base=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ConvBlock(base*8, base*8)

        # Decoder
        self.up4  = nn.ConvTranspose2d(base*8, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*8 + base*8, base*4)

        self.up3  = nn.ConvTranspose2d(base*4, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*4 + base*4, base*2)

        self.up2  = nn.ConvTranspose2d(base*2, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*2 + base*2, base)

        self.up1  = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.dec1 = ConvBlock(base + base, base)

        self.final = nn.Sequential(
            nn.Conv2d(base, 3, 1),
            nn.Sigmoid()   # Output in [0, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._up(self.up4, bn, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(self.up3, d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(self.up2, d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(self.up1, d2, e1), e1], dim=1))

        return self.final(d1)

    @staticmethod
    def _up(upsample, x, skip):
        """Upsample and fix size mismatch from odd dimensions."""
        x = upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return x