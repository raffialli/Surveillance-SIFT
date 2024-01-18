import cv2
import os
import numpy as np
import shutil

# Function to calculate the difference between frames
def calculate_difference(frame1, frame2):
  
    difference = cv2.absdiff(frame1, frame2)
    sum_difference = np.sum(difference)
    return sum_difference

# Paths to the directories
input_directory = r"E:\\TensorFlow\\revised dataset\\all-non-human"
moved_directory = r"E:\\TensorFlow\\revised dataset\\moved-non-human"
final_directory = r"E:\\TensorFlow\\revised dataset\\final-non-human"

# Initialize variables
prev_frame = None

# Loop through each file in the input directory
for filename in sorted(os.listdir(input_directory)):
    filepath = os.path.join(input_directory, filename)
    
    # Read the current frame
    current_frame = cv2.imread(filepath, 0)
    
    # If there's a previous frame, calculate the difference
    if prev_frame is not None:
        diff = calculate_difference(prev_frame, current_frame)
        
         # Log the difference to a file without stopping the script
        with open(r"C:\Code\Synology\Pytorch\differences_log.txt", "a") as log_file:

            log_file.write(f"{diff}\n")
            print(os.getcwd())
        
        # If the difference is <= 18,000,000, move to 'moved' folder
        if diff <= 100000:
            shutil.copy(filepath, os.path.join(moved_directory, filename))
        
        # If the difference is > 18,000,000, move to 'final' folder
        else:
            shutil.copy(filepath, os.path.join(final_directory, filename))
            print("moving to final folder")

    # Update the previous frame for the next iteration
    prev_frame = current_frame
