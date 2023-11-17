import torch
import torch.nn as nn
# from torchsummary import summary
from model import VRC
import time
from fvcore.nn import FlopCountAnalysis
import torch.autograd.profiler as profiler
# # $ flops = FlopCountAnalysis(model, input)
# # $ flops.total()

model = VRC()
inputs = torch.randn(1, 15, 3, 224, 224)
model.eval()

# # summary(model, input_size=(15, 3, 224, 224), device="cpu")

# # flops = FlopCountAnalysis(model, inputs)
# # print("The total number of FLOPs is: ", flops.total()/1e9," GFLOPs")




# Create an instance of your model

# Set the model to evaluation mode

# Define an input tensor (replace this with your actual input data)
# input_data = torch.randn(1, 3, 224, 224)

# Use torch.autograd.profiler to profile the forward pass
with profiler.profile(record_shapes=True, use_cuda=False) as prof:
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(inputs)

# Print the profile results
print(prof.key_averages().table(sort_by="self_cpu_time_total"))


def get_time(model, inputs, epochs=100, warmup=10):

    model = torch.jit.script(model)
    for i in range(warmup):
        preds = model(inputs)
    start_time = time.perf_counter()
    for i in range(epochs):
        preds = model(inputs)
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time)/epochs
    return total_time

print("The total time is: ", get_time(model, inputs), "s")
