import sys
import os
from PIL import Image
from tqdm import tqdm

file_list = os.listdir("Yoga-82/yoga_dataset_links");

f = open("Yoga-82/black_list.txt", 'r')
black_list = f.readlines();
f.close();
black_list = [txt.strip() for txt in black_list]

try:
    os.mkdir("Images")
    print('Folder "./Images" created.')
except:
    print('Folder "./Images" already exists.')

for i in tqdm(range(len(file_list))):
    if(file_list[i][-4:]=='.txt'):
        f = open("Yoga-82/yoga_dataset_links/" + file_list[i], 'r');
        lines = f.readlines();
        f.close();
        
        with open('Yoga-82/black_list.txt', 'a') as file:
            for j in tqdm(range(len(lines))):
                splits = lines[j][:len(lines[j])-1].split()
                img_path = splits[0];
                link = splits[1];
                folder_name, img_name = img_path.split("/");
                if(not "Images/" + folder_name + "/" + img_name in black_list):
                    if(j == 0):
                        if(not os.path.isdir("Images/" + folder_name)):
                            os.mkdir("Images/" + folder_name)
                    if(not os.path.isfile("Images/" + folder_name + "/" + img_name)):
                        os.system("wget -q -t 1 -O " + "Images/" + folder_name + "/" + img_name + " " + link)
                    try:
                        img = Image.open("Images/" + folder_name + "/" + img_name)
                        img.verify()   # to veify if its an img
                        img.close()     #to close img and free memory space
                    except (IOError, SyntaxError) as e:
                        os.remove("Images/" + folder_name + "/" + img_name) #Remove corrupted files
                        file.write("Images/" + folder_name + "/" + img_name + "\n")