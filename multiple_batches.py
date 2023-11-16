import os
import time
files = os.listdir("configs/VRC")

all_commands = [
    "configs/VRC/C2D/C2D_8x8_R50_0.1_fixed.yaml",
    "configs/VRC/I3D/I3D_8x8_R50_0.5.yaml",
    "configs/VRC/X3D_XS/X3D_XS_0.05_fixed.yaml",
    "configs/VRC/SLOW/SLOW_8x8_R50.yaml",
    "configs/VRC/SLOWFAST/SLOWFAST_8x8_R50_0.1_fixed.yaml",
]
for command in all_commands:
    if not os.path.exists(command):
        print(f"command {command} does not exist")

all_commands = [f"sbatch submit_debug.sh {x}" for x in all_commands]        
# for file in files:
#     if file.endswith("yaml"):
#         continue
#     configs = os.listdir("configs/VRC/" + file)
#     for config in configs:
#         if "debug" not in config:
#             continue
#         string = f"python tools/run_net.py --cfg configs/VRC/{file}/{config}"
#         all_commands.append(string)
        
for command in all_commands:
    print(command)
    os.system(command)
#     time.sleep(120)