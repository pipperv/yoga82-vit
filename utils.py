import os

import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

import vit

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
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = len(image_paths)

    for image_path in tqdm(image_paths):
        image = read_image(image_path,ImageReadMode.RGB).to(float) / 255.0

        current_mean = image.mean(dim=(1, 2))
        current_std = image.std(dim=(1, 2))
        if not np.isnan(current_mean.numpy()).any():
            mean += current_mean
        if not np.isnan(current_std.numpy()).any():
            std += current_std

    mean /= total_images
    std /= total_images

    return mean, std

def load_from_file(file_name="Yoga-82/dataset.txt",print_total_dataset=False):
    file = open(file_name, 'r')
    lst = file.readlines()
    file.close()
    lst = [txt.strip().split(' ') for txt in lst]
    dataset, labels = zip(*lst)
    labels = [int(label) for label in labels]
    if print_total_dataset:
        print(f"Dataset Total: {len(dataset)}")
    return dataset, labels

def get_train(file_name="Yoga-82/yoga_train.txt",img_path='Images/'):
    file = open(file_name, 'r')
    lst = file.readlines()
    file.close()
    lst = [txt.strip().split(',') for txt in lst]
    lst = [[img_path+"_".join(item[0].split(' ')),int(item[1]),int(item[2]),int(item[3])] for item in lst]
    dataset, _ = load_from_file()
    final_list = []
    for item in lst:
        if item[0] in dataset:
            final_list.append(item)
    print(f"Train Dataset Total: {len(final_list)}")
    images, labels_6, labels_20, labels_82 = zip(*final_list)
    return list(images), list(labels_6), list(labels_20), list(labels_82)

def get_test(file_name="Yoga-82/yoga_test.txt",img_path='Images/'):
    file = open(file_name, 'r')
    lst = file.readlines()
    file.close()
    lst = [txt.strip().split(',') for txt in lst]
    lst = [[img_path+"_".join(item[0].split(' ')),int(item[1]),int(item[2]),int(item[3])] for item in lst]
    dataset, _ = load_from_file()
    final_list = []
    for item in lst:
        if item[0] in dataset:
            final_list.append(item)
    print(f"Test Dataset Total: {len(final_list)}")
    images, labels_6, labels_20, labels_82 = zip(*final_list)
    return list(images), list(labels_6), list(labels_20), list(labels_82)

def get_class_list(directory_path='Images'):
    classes = os.listdir(directory_path)
    print(f"Total Classes: {len(classes)}")
    return classes

def create_from_pretrained(config, path='ViT-B_16.npz'):
    # Create a model
    model = vit.ViT(config).to('cuda')
    # Load pretrained file
    if path == 'ViT-B_16.npz':
        npzfile = np.load(path)
        # Create a custom State Dict Loader for key mismatching
        names, _ = zip(*model.named_parameters())
        for key in npzfile.files:
            new_key = key.replace('/','.').replace('scale','weight').replace('kernel','weight')
            new_key = new_key.replace('encoderblock_','encoderblock.')
            if key == 'head/bias' or key == 'head/kernel':
                pass
            elif new_key in names:
                try:
                    model.state_dict()[new_key].copy_(torch.Tensor(npzfile[key]))
                except:
                    model.state_dict()[new_key].copy_(torch.Tensor(npzfile[key].transpose()))
            elif ('query' in new_key or 'key' in new_key or 'value' in new_key):
                if 'weight' in new_key:
                    new_key = new_key.replace('_','.').split('.')
                    n_block = new_key[2]
                    qkv = new_key[-2]
                    for i, param in enumerate(torch.Tensor(npzfile[key].transpose(1, 2, 0))):
                        model.state_dict()[f'Transformer.encoderblock.{n_block}.MultiHeadDotProductAttention_1.heads.{i}.{qkv}.weight'].copy_(param)
                if 'bias' in new_key:
                    new_key = new_key.replace('_','.').split('.')
                    n_block = new_key[2]
                    qkv = new_key[-2]
                    for i, param in enumerate(torch.Tensor(npzfile[key])):
                        model.state_dict()[f'Transformer.encoderblock.{n_block}.MultiHeadDotProductAttention_1.heads.{i}.{qkv}.bias'].copy_(param)
            elif 'out' in new_key:
                if 'weight' in new_key:
                    new_key = new_key.replace('_','.').split('.')
                    n_block = new_key[2]
                    param = torch.Tensor(npzfile[key].transpose())
                    n_0, n_1, n_2 = param.size()
                    param = torch.reshape(param, (n_0, n_1 * n_2))
                    model.state_dict()[f'Transformer.encoderblock.{n_block}.MultiHeadDotProductAttention_1.output_linear.weight'].copy_(param)
                if 'bias' in new_key:
                    param = torch.Tensor(npzfile[key].transpose())
                    model.state_dict()[f'Transformer.encoderblock.{n_block}.MultiHeadDotProductAttention_1.output_linear.weight'].copy_(param)
            elif 'out' in new_key:
                model.state_dict()[new_key].copy_(torch.Tensor(npzfile[key].transpose()))
            elif 'embedding' in new_key:
                if 'weight' in new_key:
                    model.state_dict()['embbeding.projection.weight'].copy_(torch.Tensor(npzfile[key].transpose()))
                if 'bias' in new_key:
                    model.state_dict()['embbeding.projection.bias'].copy_(torch.Tensor(npzfile[key].transpose()))
            elif new_key == 'cls': 
                model.state_dict()['token_embedding.token'].copy_(torch.Tensor(npzfile[key]))
            else:
                print(new_key, torch.Tensor(npzfile[key]).size())
        
        return model