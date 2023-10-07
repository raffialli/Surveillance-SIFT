import cv2
import os
from PIL import Image
import shutil
from random import shuffle

# Define paths with raw string literals
video_folder = r'E:\TensorFlow\Front Door Vids\human'
output_folder = r'E:\TensorFlow\Front Door Vids\human\images'
backup_folder = r'E:\TensorFlow\backup\training_images\human'
resize_folder = r'E:\TensorFlow\backup\training_images\resize-human'
train_dir = r'E:\TensorFlow\Front Door Vids\training-files\training\human'
val_dir = r'E:\TensorFlow\Front Door Vids\training-files\validation\human'

# Video to Images
all_files = os.listdir(video_folder)
video_files = [f for f in all_files if f.endswith('.mp4')]

os.makedirs(output_folder, exist_ok=True)

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_filename = os.path.join(output_folder, f"{video_file[:-4]}_frame_{frame_number}.jpg")
        cv2.imwrite(output_filename, frame)

        frame_number += 1

    cap.release()

print("Done extracting frames.")

# Backup and Resize Images
os.makedirs(backup_folder, exist_ok=True)
os.makedirs(resize_folder, exist_ok=True)

new_size = (320, 180)

for filename in os.listdir(output_folder):
    print(f"Processing {filename}...")
    try:
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            img_path = os.path.join(output_folder, filename)

            # Move to backup folder
            shutil.move(img_path, os.path.join(backup_folder, filename))
            print(f"Moved {filename} to backup folder.")

            # Open from backup folder for resizing
            img = Image.open(os.path.join(backup_folder, filename))
            img_resized = img.resize(new_size, 3)

            # Save to resize folder
            img_resized_path = os.path.join(resize_folder, filename)
            img_resized.save(img_resized_path)
            print(f"Saved resized image to {img_resized_path}")

        else:
            print(f"Skipping {filename}, not an image.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Resizing completed.")

# Move Images to Training and Validation Sets
all_files = [f for f in os.listdir(resize_folder) if os.path.isfile(os.path.join(resize_folder, f))]
shuffle(all_files)

num_train = int(len(all_files) * 0.9)
num_val = len(all_files) - num_train

for i, file_name in enumerate(all_files):
    if i < num_train:
        shutil.move(os.path.join(resize_folder, file_name), os.path.join(train_dir, file_name))
    else:
        shutil.move(os.path.join(resize_folder, file_name), os.path.join(val_dir, file_name))

print(f"Moved {num_train} files to the training set.")
print(f"Moved {num_val} files to the validation set.")
