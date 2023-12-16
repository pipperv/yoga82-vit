import os

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

def list_images(directory_path):
    dir_list = os.listdir(directory_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    classes = dir_list
    image_files = []
    for directory in dir_list:
        img_dir = directory_path+"/"+directory
        dir_files = [img_dir+"/"+file for file in os.listdir(img_dir) if any(file.lower().endswith(ext) for ext in image_extensions)]
        image_files.append(dir_files)
    return image_files, classes

def verify_img(img_dir):
    try:
        #img = Image.open(img_dir)
        #img.verify()
        #img.close()
        read_image(img_dir,ImageReadMode.RGB)
        return True
    except:
        return False

def verify_dataset(directory_path='Images', show_valid_per_class=False, write_file=True):    
    directory_path = directory_path
    image_list, classes = list_images(directory_path)

    total_imgs = 0
    total_valid_imgs = 0
    valid_dataset = []
    labels = []

    with open('Yoga-82/dataset.txt', 'w') as file:
        for class_imgs in tqdm(image_list):
            total_class_imgs = len(class_imgs)
            total_valid_class_imgs = 0
            class_name = class_imgs[0].split("/")[1]
            for img_name in class_imgs:
                total_imgs+=1
                if verify_img(img_name):
                    total_valid_class_imgs+=1
                    total_valid_imgs+=1
                    valid_dataset.append(img_name)
                    class_name = img_name.split('/')[1]
                    label = classes.index(class_name)
                    labels.append(label)
                    if write_file:
                        file.write(img_name + ' ' + str(label) + "\n")
                else:
                    pass
            if show_valid_per_class:
                print(f"{class_name}: {total_valid_class_imgs}/{total_class_imgs}")

    print(f"Dataset Total: {total_valid_imgs}/{total_imgs}")
    return valid_dataset, labels, classes

def calculate_mean_std(image_paths):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = len(image_paths)

    for image_path in tqdm(image_paths):
        image = read_image(image_path,ImageReadMode.RGB)
        image = image.to(float) / 255

        mean += image.mean(dim=(1, 2)).item()
        std += image.std(dim=(1, 2)).item()

    mean /= total_images
    std /= total_images

    return mean, std

def load_from_file(file_name="Yoga-82/dataset.txt", show_valid_per_class=False):
    
    file = open(file_name, 'r')
    lst = file.readlines()
    file.close()
    lst = [txt.strip().split(' ') for txt in lst]
    dataset, labels = zip(*lst)
    labels = [int(label) for label in labels]

    print(f"Dataset Total: {len(dataset)}")
    return dataset, labels

def get_class_list(directory_path='Images'):
    classes = os.listdir(directory_path)
    print(f"Total Classes: {len(classes)}")
    return classes