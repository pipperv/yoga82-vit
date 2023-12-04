import sys
import os
import argparse
from PIL import Image
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='List all images in a directory.')
    parser.add_argument('-b','--use-black-list', default=False, help='Uses the Black List to skip unavailable links.')
    parser.add_argument('-r','--remove-corrupted', default=False, help='Removes the file after download if fails verification.')
    parser.add_argument('-t','--tries', default=3, help='Times the files will be tried to download.')
    parser.add_argument('-w','--write-blacklist', default=True, help='Black List will be generated.')
    parser.add_argument('--timeout', default=5, help='Timeout value.')
    parser.add_argument('-q','--quiet', default=True, help='wget will not show output.')
    
    file_list = os.listdir("Yoga-82/yoga_dataset_links");

    args = parser.parse_args()

    black_listed = args.use_black_list
    remove_corrupted = args.remove_corrupted
    tries = args.tries
    write_black_list = args.write_blacklist
    timeout = args.timeout
    quiet = args.quiet

    print(tries)
    print(timeout)
    print(black_listed)

    if quiet:
        q = ' -q '
    else:
        q = ' '

    if black_listed:
        f = open("Yoga-82/black_list.txt", 'r')
        black_list = f.readlines();
        f.close();
        black_list = [txt.strip() for txt in black_list]
    else:
        black_list = []

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
                            os.system(f"wget{q}--tries {tries} -O " + "Images/" + folder_name + "/" + img_name + " " + link)
                        try:
                            img = Image.open("Images/" + folder_name + "/" + img_name)
                            img.verify()   # to veify if its an img
                            img.close()     #to close img and free memory space
                        except (IOError, SyntaxError) as e:
                            if remove_corrupted:
                                os.remove("Images/" + folder_name + "/" + img_name) #Remove corrupted files
                            if write_black_list:
                                file.write("Images/" + folder_name + "/" + img_name + "\n")

if __name__ == "__main__":
    main()