import os
import shutil
from random import shuffle

# Define the original directory where the resized images are
original_dir = 'E:\\TensorFlow\\backup\\training_images\\resize-human'

# Define destination directories for training and validation
train_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\training\\human'
val_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\validation\\human'

# List all files in the original directory
all_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]

# Shuffle files randomly
shuffle(all_files)

# Calculate the number of files for training and validation
num_train = int(len(all_files) * 0.9)
num_val = len(all_files) - num_train

# Move files to training and validation directories
for i, file_name in enumerate(all_files):
    if i < num_train:
        shutil.move(os.path.join(original_dir, file_name), os.path.join(train_dir, file_name))
    else:
        shutil.move(os.path.join(original_dir, file_name), os.path.join(val_dir, file_name))

print(f"Moved {num_train} files to the training set.")
print(f"Moved {num_val} files to the validation set.")
