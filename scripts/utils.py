import os
import PIL
from PIL import Image
from shutil import copyfile
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv


'''
Resize all images to 30x30 pixels
'''
def resize(in_file, out_file):

    baseheight = 31
    img = Image.open(in_file)
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)

    baseheight = 30
    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)

    img.save(out_file)

'''
Take copy of training and testing data and remove every nth image.
This is to prepare data of test accuracy of different training sample sizes.
'''
def prepare_varying_sample_sizes(divisor, path):

    dir = os.listdir(path)

    i = 0 
    for file in dir:
        if(i%divisor == 0):
            os.remove(os.path.join(path,file))
            i += 1
        else:
            i += 1

'''
Get RGB frequencies for an image
'''
def get_RGB_freq(path, freq):

    photo = Image.open(path) #your image
    photo = photo.convert('RGB')

    width = photo.size[0] #define W and H
    height = photo.size[1]

    for y in range(0, height): #each pixel has coordinates
        for x in range(0, width):
            RGB = photo.getpixel((x,y))
            r,g,b = RGB
            freq[r][0] += 1
            freq[g][1] += 1
            freq[b][2] += 1
    return freq

'''
Write RGB frequencies to excel
'''
def write_csv(freq):
    myFile = open('roof.csv', 'w')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(freq)


'''
# resize
in_file_root = "C:/Users/Anna Ptasznik/Desktop/pool_finder/"
out_file_root = "C:/Users/Anna Ptasznik/Desktop/resized_training"
paths = ["pool", "lawn", "roof", "street" ]

for path in paths:
    in_file_path = os.path.join(in_file_root, path)
    dir = os.listdir(in_file_path)

    for file in dir:
        in_file = os.path.join(in_file_path,file)
        out_file = os.path.join(out_file_root, path, file)
        resize(in_file, out_file)
'''