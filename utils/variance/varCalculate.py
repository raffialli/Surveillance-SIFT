# this code was written with Advanced Data Analysis turned on in GPT-4
import sys
print(sys.executable)
import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

# Function to compare two images using SSIM
def compare_images(imageA, imageB, filenameA, filenameB):
    s = ssim(imageA, imageB)
    return (filenameA, filenameB, s)

try:
    input_directory = "E:\\TensorFlow\\revised dataset\\all-non-human"
    filenames = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.png'))][:10]  # Limit to first 10 for testing
    frames = [cv2.imread(os.path.join(input_directory, f), cv2.IMREAD_GRAYSCALE) for f in filenames]

    # Prepare for parallel processing of image comparison
    differences = []
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers to your CPU
        futures = []
        for i in range(len(frames)):
            for j in range(i+1, len(frames)):
                futures.append(executor.submit(compare_images, frames[i], frames[j], filenames[i], filenames[j]))

        for future in futures:
            differences.append(future.result())

    # Log differences
    with open("E:\\TensorFlow\\revised dataset\\diff_log.txt", "a") as log_file:
        for filename1, filename2, difference in differences:
            log_file.write(f"{filename1} vs {filename2}: {difference}\n")

    avg_difference = np.mean([diff for _, _, diff in differences])
    print(f"Average SSIM difference between frames: {avg_difference}")

except Exception as e:
    print(f"An error occurred: {e}")
