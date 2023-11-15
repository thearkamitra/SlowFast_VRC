import torch
import torch.nn as nn
from torchsummary import summary
from model import VRC


model = VRC()
inputs = torch.randn(1, 15, 3, 224, 224)

summary(model, input_size=(15, 3, 224, 224), device="cpu")
