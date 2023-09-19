from PIL import Image
import os

input_folder = "E:\TensorFlow\Front Door Vids\human\images"
new_size = (128, 128)

for filename in os.listdir(input_folder):
    print(f"Processing {filename}...")
    try:
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            img_path = os.path.join(input_folder, filename)
            print(f"Opening {img_path}")
            img = Image.open(img_path)
            img_resized = img.resize(new_size, 3)
            
            print(f"Replacing original image {img_path}")
            img_resized.save(img_path)
        else:
            print(f"Skipping {filename}, not an image.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Resizing completed.")
