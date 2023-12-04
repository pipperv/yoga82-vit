import sys
import os
from PIL import Image
from tqdm import tqdm

def list_images(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_files = [file for file in os.listdir(directory) if any(file.lower().endswith(ext) for ext in image_extensions)]
    return image_files

def verify_img(img_dir):
    try:
        img = Image.open(img_dir)
        img.verify()
        img.close()
        return True
    except:
        return False

# Replace 'your_directory_path' with the actual path to your directory
directory_path = 'Images'
image_list = list_images(directory_path)

print(len(image_list))