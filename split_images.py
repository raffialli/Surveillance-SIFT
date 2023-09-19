import os
import shutil
from random import shuffle

# Define source and destination directories
src_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\training\\non-human'
val_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\validation\\non-human'
test_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\testing\\non-human'

# List all files in the source directory
all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

# Shuffle files randomly
shuffle(all_files)

# Calculate the number of files for training, validation, and testing
num_train = int(len(all_files) * 0.8)
num_val = int(len(all_files) * 0.1)
num_test = len(all_files) - num_train - num_val  # Alternatively, int(len(all_files) * 0.1)

# Move files
for i, file_name in enumerate(all_files):
    if i < num_train:
        continue  # Skip files already in training
    elif i < num_train + num_val:
        shutil.move(os.path.join(src_dir, file_name), os.path.join(val_dir, file_name))
    else:
        shutil.move(os.path.join(src_dir, file_name), os.path.join(test_dir, file_name))
