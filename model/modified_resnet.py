import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlockIdentity(nn.Module):

    def __init__(self, input_depth, f1, f2):
        super(ResidualBlockIdentity, self).__init__()
       
        self.res_block_identity = nn.Sequential(
            nn.Conv2d(input_depth, f1, kernel_size=(1,1), stride=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f1, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=(1,1), stride=1, bias=False),
            nn.BatchNorm2d(f2)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.res_block_identity(x)
        out += x
        out = self.relu(out)
        return out


class ResidualBlockConvolution(nn.Module):

    def __init__(self, input_depth, s, f1, f2):
        super(ResidualBlockConvolution, self).__init__()
        
        self.res_block_conv_part1 = nn.Sequential(
            nn.Conv2d(input_depth, f1, kernel_size=(1,1), stride=s, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f1, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=(1,1), stride=1, bias=False)
        )
        
        self.res_block_conv_part2 = nn.Conv2d(input_depth, f2, kernel_size=(1,1), stride=s, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        out1 = self.res_block_conv_part1(x)
        out2 = self.res_block_conv_part2(x)
    
        out = out1 + out2
        out = self.relu(out)

        return out
    
    
class ModifiedResNet(nn.Module): # input size 256x256
    def __init__(self, n_keypoints=21):
        super(ModifiedResNet, self).__init__()
        
        self.conv_pool_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7,7), stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2)
        )
        self.res_block1 = nn.Sequential(
            ResidualBlockConvolution(64, s=1, f1=64, f2=256),
            ResidualBlockIdentity(256, f1=64, f2=256),
            ResidualBlockIdentity(256, f1=64, f2=256)
        )
        self.res_block2 = nn.Sequential(
            ResidualBlockConvolution(256, s=2, f1=128, f2=512),
            ResidualBlockIdentity(512, f1=128, f2=512),
            ResidualBlockIdentity(512, f1=128, f2=512)
        )
        self.res_block3 = nn.Sequential(
            ResidualBlockConvolution(512, s=2, f1=256, f2=1024),
            ResidualBlockIdentity(1024, f1=256, f2=1024),
            ResidualBlockIdentity(1024, f1=256, f2=1024)
        )        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=(3,3), padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.keypoints_2d_prediction = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(3,3), padding=1, stride=1, bias=False),
            nn.ConvTranspose2d(64, n_keypoints, kernel_size=(4,4), stride=2, bias=False)
        ) 
        
        self.keypoints_3d_prediction = nn.Sequential(
            nn.Linear(256*31*31, 200, bias=True),  # dimensionality is hard coded
            nn.Linear(200, 3*n_keypoints, bias=True)
        ) 
        
    def forward(self, sample):
        features = self.conv_pool_block(sample['image'])
        features = self.res_block1(features)
        features = self.res_block2(features)
        features = self.res_block3(features)
        features = self.conv_block(features)
        
        out_3d = self.keypoints_3d_prediction(features.view(features.size(0), -1))
        out_2d = self.keypoints_2d_prediction(features)
        del(features)
        out_2d = F.interpolate(out_2d, scale_factor=4, mode='bilinear', align_corners=True)
        
        
        return {'heatmap_2d': out_2d, 'vector_3d': out_3d}
    
    
