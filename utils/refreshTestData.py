import os
import shutil

# Source and destination directories
source_dir = r'E:\TensorFlow\backup\20230301PM'
destination_dir = r'E:\TensorFlow\20230301PM'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through files in the source directory (not subdirectories)
for file in os.listdir(source_dir):
    source_path = os.path.join(source_dir, file)
    destination_path = os.path.join(destination_dir, file)

    # Check if the file is a video file (you can customize this check)
    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Check if the file already exists in the destination directory
        if not os.path.exists(destination_path):
            # Copy the file from source to destination
            shutil.copy2(source_path, destination_path)

