import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

# 기존 모델 : upsampling을 통해 원본 이미지와 동일하게 복원
# input image size : 256x256 , nb_classes= 3
# 여기서는 EfficientNet을 Backbone, Upsampling으로 size 복원
# 그러나 크기는 224x224로 변환후 학습 진행 예정(pretrain된 모델 사용하므로)

from torchvision.models import efficientnet_b0

class EfficientNet(nn.Module):
    def __init__(self, nb_classes, nb_filters = 16):
        # 논문에서 진행한 nb_classes= 3 / nb_filters=16
        super(EfficientNet, self).__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        # input data는 channel이 1이므로 input channel의 변경이 필수적 / 나머지는 동일하게 진행
        #  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder = self.backbone.features

        self.upsample1 = self.upsample_block(1280,640)
        self.conv1 = self.ConvBlock(640,320)

        self.upsample2 = self.upsample_block(320,160)
        self.conv2 = self.ConvBlock(160,80)
        
        self.upsample3 = self.upsample_block(80,40)
        self.conv3 = self.ConvBlock(40,20)
        
        self.upsample4 = self.upsample_block(20,nb_filters)
        self.conv4 = self.ConvBlock(nb_filters,nb_filters)
        
        self.upsample5 = self.upsample_block(nb_filters, nb_filters)
        self.conv5 = self.ConvBlock(nb_filters, nb_filters)
        self.conv6 = nn.Conv2d(nb_filters, nb_classes, kernel_size=1, stride=1, padding=0)

    def upsample_block(self,input_channels,output_channels,kernel_size=1,stride=1,padding=0):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        
    def ConvBlock(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self,x):
        x = self.encoder(x)

        x = self.upsample1(x)  
        x = self.conv1(x)

        x = self.upsample2(x)
        x = self.conv2(x)

        x = self.upsample3(x)
        x = self.conv3(x)

        x = self.upsample4(x)
        x = self.conv4(x)

        x = self.upsample5(x)
        x = self.conv5(x)

        x = self.conv6(x)
        output = F.log_softmax(x, dim=1)
        return output
        
