import torch
import torch.nn as nn
from torchvision.models import mobilenet


class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride_dw=1, stride_pw=1, consider_pw = True):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride_dw, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride_pw, padding=0)
        self.batchnorm_dw = nn.BatchNorm2d(in_channels)
        self.batchnorm_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.consider_pw = consider_pw
    def forward(self, x):
        x = self.depthwise(x)
        x = self.batchnorm_dw(x)
        x = self.relu(x)
        if not self.consider_pw:
            return x
        x = self.pointwise(x)
        x = self.batchnorm_pw(x)
        x = self.relu(x)
        return x

class MobileNet(nn.Module):
    def __init__(self,  alpha=1.0,  full_layer = False):
        super(MobileNet, self).__init__()
        self.first_layer =   nn.Sequential( nn.Conv2d(3, int(32*alpha), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(32*alpha)),
            nn.ReLU(),
        )
        self.consecutive_layers = nn.Sequential(
               DepthwiseSeperableConv(int(32*alpha), int(64*alpha), stride_dw=1, stride_pw=1),
                DepthwiseSeperableConv(int(64*alpha), int(128*alpha), stride_dw=2, stride_pw=1),
                DepthwiseSeperableConv(int(128*alpha), int(128*alpha), stride_dw=1, stride_pw=1),
                DepthwiseSeperableConv(int(128*alpha), int(256*alpha), stride_dw=2, stride_pw=1),
                DepthwiseSeperableConv(int(256*alpha), int(256*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(256*alpha), int(512*alpha), stride_dw=2),
                DepthwiseSeperableConv(int(512*alpha), int(512*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(512*alpha), int(512*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(512*alpha), int(512*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(512*alpha), int(512*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(512*alpha), int(512*alpha), stride_dw=1),
                DepthwiseSeperableConv(int(512*alpha), int(1024*alpha), stride_dw=2),
        )
        self.full_layer = full_layer
        if self.full_layer:
            self.last_layer = DepthwiseSeperableConv(int(1024*alpha), int(1024*alpha), stride_dw=2)
        else:
            self.last_layer = DepthwiseSeperableConv(int(1024*alpha), int(1024*alpha), stride_dw=2, consider_pw=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d((1,1))    
        
    def forward(self,x):
        x = self.first_layer(x)
        x = self.consecutive_layers(x)
        x = self.last_layer(x)
        x = self.pooling_layer(x)
        return x

class VRC(nn.Module):
    def __init__(self, alpha=1.0, num_classes = 13, full_layer = True, num_frames = 15, dropout = 0.25):
        super(VRC, self).__init__()
        self.mobilenet_transfer = MobileNet(alpha=alpha, full_layer=full_layer, )
        self.GRU = nn.GRU(input_size=1024, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout2 = nn.Dropout(p=dropout)
        self.relu2 = nn.ReLU()
        self.num_frames = num_frames
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.mobilenet_transfer(x)
        x = x.view(B, T, -1)
        _, x = self.GRU(x)
        x = x.view(B, -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        return x

if __name__ == "__main__":
    model = VRC()
    print(model)
    x = torch.randn(2, 15, 3, 224, 224)
    y = model(x)
    print(y.shape)
    
