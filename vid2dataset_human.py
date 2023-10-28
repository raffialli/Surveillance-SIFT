# Import necessary libraries
import cv2  # OpenCV library for video processing
import os   # Operating system interaction
from PIL import Image  # Python Imaging Library for image manipulation
import shutil  # File and directory operations
from random import shuffle  # Randomize the order of files

# Define paths with raw string literals
video_folder = r'E:\TensorFlow\Front Door Vids\human'  # Folder containing video files
output_folder = r'E:\TensorFlow\Front Door Vids\human\images'  # Folder to store extracted frames as images
backup_folder = r'E:\TensorFlow\backup\training_images\human'  # Folder to backup original images
resize_folder = r'E:\TensorFlow\backup\training_images\resize-human'  # Folder to store resized images
train_dir = r'E:\TensorFlow\Front Door Vids\training-files\training\human'  # Folder for training images
val_dir = r'E:\TensorFlow\Front Door Vids\training-files\validation\human'  # Folder for validation images

# Video to Images
# List all files in the video folder
all_files = os.listdir(video_folder)
# Filter out files with .mp4 extension
video_files = [f for f in all_files if f.endswith('.mp4')]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each video file
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    # Check if the video can be opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the output filename for each frame
        output_filename = os.path.join(output_folder, f"{video_file[:-4]}_frame_{frame_number}.jpg")
        # Save the frame as an image
        cv2.imwrite(output_filename, frame)

        frame_number += 1
    
    cap.release()

print("Done extracting frames.")

# Backup and Resize Images
# Create backup and resize folders if they don't exist
os.makedirs(backup_folder, exist_ok=True)
os.makedirs(resize_folder, exist_ok=True)

# Define the new size for resizing
new_size = (320, 180)

# Loop through each file in the output folder
for filename in os.listdir(output_folder):
    print(f"Processing {filename}...")
    try:
        # Check if the file is a JPG or PNG image
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            img_path = os.path.join(output_folder, filename)
            
            # Move the image to the backup folder
            shutil.move(img_path, os.path.join(backup_folder, filename))
            print(f"Moved {filename} to backup folder.")

            # Open the image from the backup folder for resizing
            img = Image.open(os.path.join(backup_folder, filename))
            img_resized = img.resize(new_size, 3)

            # Save the resized image to the resize folder
            img_resized_path = os.path.join(resize_folder, filename)
            img_resized.save(img_resized_path)
            print(f"Saved resized image to {img_resized_path}")

        else:
            print(f"Skipping {filename}, not an image.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Resizing completed.")

# Move Images to Training and Validation Sets
# List all files in the resize folder
all_files = [f for f in os.listdir(resize_folder) if os.path.isfile(os.path.join(resize_folder, f))]
shuffle(all_files)

# Calculate the number of files for training and validation
num_train = int(len(all_files) * 0.9)
num_val = len(all_files) - num_train

# Move files to training and validation directories
for i, file_name in enumerate(all_files):
    if i < num_train:
        shutil.move(os.path.join(resize_folder, file_name), os.path.join(train_dir, file_name))
    else:
        shutil.move(os.path.join(resize_folder, file_name), os.path.join(val_dir, file_name))

print(f"Moved {num_train} files to the training set.")
print(f"Moved {num_val} files to the validation set.")
