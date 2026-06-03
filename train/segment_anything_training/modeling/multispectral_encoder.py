import torch
import torch.nn as nn
# from image_encoder import ImageEncoderViT

class MultispectralEncoder_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(MultispectralEncoder_Conv, self).__init__()
        
        padding = (kernel_size - 1) // 2 # 保持空间不变

        # Depthwise: 每个输入通道对应一个卷积核（group=in_channels）
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels)
        )
        
        # Pointwise: 1x1 卷积进行通道混合
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

#TODO 
class MultispectralEncoder_ViT(nn.Module):
    pass

class FeatureCompressor(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),   # 512x512
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),            # 256x256
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),           # 128x128
            nn.ReLU(),
            nn.Conv2d(256, out_channels, 3, stride=2, padding=1),  # 64x64
        )

    def forward(self, x):
        return self.encoder(x)  # [1, 256, 64, 64]
    
class FeatureCompressor_linear(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4, padding=0),    # 1024 → 256
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),            # 256 → 128
            nn.Conv2d(128, out_channels, kernel_size=2, stride=2, padding=0)   # 128 → 64
        )

    def forward(self, x):
        return self.compressor(x)  # [B, 256, 64, 64]
    
class CBAMUNet(nn.Module):
    def __init__(self,in_channels,out_channels=3,if_to_RGB=False,if_linear_proj=False):
        super(CBAMUNet, self).__init__()

        # 编码器
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)

        # 解码器
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(128 + 64, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # Back To ms Layer
        if if_to_RGB:
            if if_linear_proj:
                self.prj_to_ms = nn.Conv2d(out_channels, in_channels, kernel_size=1)
                self.vit_align = FeatureCompressor(out_channels, 256)
            else:
                self.prj_to_ms = nn.Sequential(
                    nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, in_channels, kernel_size=1)
                )
                self.vit_align = FeatureCompressor_linear(out_channels, 256)
        self.if_to_RGB = if_to_RGB
    def forward(self, x):
        # 编码阶段
        x1 = self.enc1(x)      # 64
        x2 = self.enc2(self.pool1(x1))  # 128
        x3 = self.enc3(self.pool2(x2))  # 256

        # 解码阶段 + skip connection
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)  # 拼接 encoder 的特征
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        out = self.out_conv(x)

        if self.if_to_RGB:
            out_ms = self.prj_to_ms(out)
            out_vit_align = self.vit_align(out)
            return out, out_ms ,out_vit_align
        else:
            return out
class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAMBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x
#CBMA copy from https://github.com/luuuyi/CBAM.PyTorch.git
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




if __name__ == '__main__':
    # test
    x = torch.randn(2, 3, 224, 224)
    model = MultispectralEncoder_Conv(3, 64)
    y = model(x)
    print(y.shape)


