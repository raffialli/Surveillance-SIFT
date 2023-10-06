"""
This script processes video files to detect human presence using a pre-trained PyTorch model.
If no human is detected in any of the frames, the video file is deleted.
"""
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from model_definition import NN  # Assuming your model is defined in this file
from PIL import Image

# Load pre-trained model
model = torch.load('FrontDoor_new_dataset_v3.pth')

# Move model to GPU if available
device = 'cuda:0'  # Change to 'cpu' if you don't have a GPU
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Define directory containing video files
folder_path = "E:\TensorFlow\\20230301PM"

# List all files in the directory
all_files = os.listdir(folder_path)

# Filter out only the .mp4 files
video_files = [f for f in all_files if f.endswith('.mp4')]

# Define the transform for frames
test_transform = transforms.Compose([
    transforms.Resize((320, 180)),
    transforms.ToTensor(),
])

# Loop over each video file
for video_file in video_files:
    video_path = os.path.join(folder_path, video_file)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Flag to track if a human is detected
    human_present = False

    # Loop over each frame in the video
    for frame_index in range(total_frames):
        ret, frame = cap.read()

        # Skip the loop iteration if the frame couldn't be read
        if not ret:
            print(f"Couldn't read frame {frame_index}. Skipping...")
            continue

        # Preprocess the frame
        frame = cv2.resize(frame, (320, 180))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
        frame_tensor = test_transform(frame)  # Apply transform

        # Prepare tensor for model prediction
        frame_tensor = frame_tensor.unsqueeze(0)
        frame_tensor = frame_tensor.to(device)
        
        # Flatten tensor to fit model input
        frame_tensor = frame_tensor.reshape(frame_tensor.shape[0], -1)
        
        # Predict using the pre-trained model
        with torch.no_grad():
            frame_tensor = frame_tensor.to(device)
            output = model(frame_tensor)
            
        # Find the most probable class
        _, predicted_class = torch.max(output.data, 1)

        # Check if human is detected (assuming human is class 0)
        if predicted_class.item() == 0:
            human_present = True
            break

    # Release video capture
    cap.release()

    # Remove video file if no human was detected
    if not human_present:
        print(f"Deleting {video_file} because no human was detected.")
        os.remove(video_path)
