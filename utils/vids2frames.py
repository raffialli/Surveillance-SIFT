#Takes vids in a folder and converts them into tensors, images, and puts them in a destination folder
#Will need to use this to extract more frames as I continue to train the model
import cv2
import os

# Define the folder containing the videos
video_folder = "E:\TensorFlow\Front Door Vids\human"
output_folder = "E:\TensorFlow\Front Door Vids\human\images"

# List all files in the folder
all_files = os.listdir(video_folder)

# Filter out the .mp4 files
video_files = [f for f in all_files if f.endswith('.mp4')]

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
