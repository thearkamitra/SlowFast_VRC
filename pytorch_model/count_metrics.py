import torch
import torch.nn as nn
from torchsummary import summary
from model import VRC
from fvcore.nn import FlopCountAnalysis
# $ flops = FlopCountAnalysis(model, input)
# $ flops.total()

model = VRC()
inputs = torch.randn(1, 15, 3, 224, 224)

summary(model, input_size=(15, 3, 224, 224), device="cpu")

flops = FlopCountAnalysis(model, inputs)
print("The total number of FLOPs is: ", flops.total()/1e9," GFLOPs")