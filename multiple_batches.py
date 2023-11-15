import os

files = os.listdir("configs/VRC")

all_commands = []
for file in files:
    if file.endswith("yaml"):
        continue
    if "SLOW" not in file:
        continue
    configs = os.listdir("configs/VRC/" + file)
    for config in configs:
        string = f"sbatch submit_debug.sh configs/VRC/{file}/{config}"
        all_commands.append(string)
        
for command in all_commands:
    os.system(command)