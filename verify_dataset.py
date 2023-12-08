import os
from PIL import Image

def list_images(directory_path):
    dir_list = os.listdir(directory_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_files = []
    for directory in dir_list:
        img_dir = directory_path+"/"+directory
        dir_files = [img_dir+"/"+file for file in os.listdir(img_dir) if any(file.lower().endswith(ext) for ext in image_extensions)]
        image_files.append(dir_files)
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

total_imgs = 0
valid_imgs = 0

for i, class_dir in enumerate(image_list):
    total_class_imgs = len(class_dir)
    valid_class_imgs = 0
    for img_dir in class_dir:
        total_imgs+=1
        if verify_img(img_dir):
            valid_class_imgs+=1
            valid_imgs+=1
        else:
            pass
    print(f"Class {i}: {valid_class_imgs}/{total_class_imgs}")

print(f"Dataset Total: {valid_imgs}/{total_imgs}")