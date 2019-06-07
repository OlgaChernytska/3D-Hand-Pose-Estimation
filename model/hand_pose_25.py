import torch
from torch import nn

from utils import latent2coords


class FirstLayer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.seq(x)


class DownLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class IntermediateLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        return self.seq(x)


class UpLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        return self.seq(x)


class LastLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(7, 7), padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 42, kernel_size=(7, 7), padding=3, bias=False),
        )

    def forward(self, x):
        return self.seq(x)


class NetworkStage(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.first_layer = FirstLayer(in_channels)

        self.down1 = DownLayer()
        self.down2 = DownLayer()
        self.down3 = DownLayer()
        self.down4 = DownLayer()
        self.down5 = DownLayer()
        self.down6 = DownLayer()

        self.intermediate_layer = IntermediateLayer()

        self.up_layer1 = UpLayer()
        self.up_layer2 = UpLayer()
        self.up_layer3 = UpLayer()
        self.up_layer4 = UpLayer()
        self.up_layer5 = UpLayer()

        self.last_layer = LastLayer()

        self.betta_layer = nn.Conv2d(42, 42, kernel_size=(1, 1), groups=42, bias=False)

    def forward(self, x):
        first_out = self.first_layer(x)

        down1 = self.down1(first_out)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)

        up1 = self.intermediate_layer(down6)
        up1 = torch.cat([up1, down5], dim=1)

        up2 = self.up_layer1(up1)
        up2 = torch.cat([up2, down4], dim=1)

        up3 = self.up_layer2(up2)
        up3 = torch.cat([up3, down3], dim=1)

        up4 = self.up_layer3(up3)
        up4 = torch.cat([up4, down2], dim=1)

        up5 = self.up_layer4(up4)
        up5 = torch.cat([up5, down1], dim=1)

        up6 = self.up_layer5(up5)
        up6 = torch.cat([up6, first_out], dim=1)

        x = self.last_layer(up6)
        x = self.betta_layer(x)
        return x


class HandPose25(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = NetworkStage(3)
        self.stage2 = NetworkStage(3 + 42)

    def forward(self, x):
        stage1 = self.stage1(x)

        x = torch.cat([x, stage1], dim=1)
        stage2 = self.stage2(x)

        coords1 = latent2coords(stage1)
        coords2 = latent2coords(stage2)
        return [coords1, coords2]
