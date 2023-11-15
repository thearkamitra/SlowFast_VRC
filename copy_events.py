import os
import shutil
import glob

# from the checkpoint files, copy the events.out.tfevents files to a new folder

checkpoint_files = os.listdir()
checkpoint_files = [f for f in checkpoint_files if f.startswith('checkpoints_')]
checkpoint_files = [f for f in checkpoint_files if "old" not in f]
new_events_dir = 'events_files_new'
os.makedirs(new_events_dir, exist_ok=True)

for checkpoint_file in checkpoint_files:
    events_folder = os.path.join(checkpoint_file, 'runs-kinetics')
    shutil.copytree(events_folder, os.path.join(new_events_dir, events_folder))

