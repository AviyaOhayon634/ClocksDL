import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicHandRemover(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = self._build_layer(3, 64)
        self.down2 = self._build_layer(64, 128)
        self.down3 = self._build_layer(128, 256)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.up_scale2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_layer2 = self._build_layer(256 + 128, 128)

        self.up_scale1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_layer1 = self._build_layer(128 + 64, 64)

        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

    def _build_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, img_tensor):
        down1 = self.down1(img_tensor)
        pool1 = self.max_pool(down1)

        down2 = self.down2(pool1)
        pool2 = self.max_pool(down2)

        down3 = self.down3(pool2)

        up2 = self.up_scale2(down3)
        if up2.size() != down2.size():
            up2 = F.interpolate(up2, size=down2.shape[2:])
        up2 = torch.cat([up2, down2], dim=1)
        up2_out = self.up_layer2(up2)

        up1 = self.up_scale1(up2_out)
        if up1.size() != down1.size():
            up1 = F.interpolate(up1, size=down1.shape[2:])
        up1 = torch.cat([up1, down1], dim=1)
        up1_out = self.up_layer1(up1)

        return torch.sigmoid(self.output_conv(up1_out))

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