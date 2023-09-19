from PIL import Image
import os
import shutil

input_folder = "E:\TensorFlow\Front Door Vids\human\images"
backup_folder = "E:\TensorFlow\\backup\\training_images\\non-human"
resize_folder = "E:\TensorFlow\\backup\\training_images\\resize-non-human"

new_size = (320, 180)

# Create backup and resize folders if they don't exist
os.makedirs(backup_folder, exist_ok=True)
os.makedirs(resize_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    print(f"Processing {filename}...")
    try:
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            
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
