# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:45:12 2017

@author: kgasb
"""
import cv2
import matplotlib.pyplot as plt


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are 
    received in RGB)
    '''
    plt.imshow(img)
    plt.title("Origin")
    plt.show()
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    #new_img = img[35:140,:,:]
    # crop to 40x320x3
    # apply subtle blur
    new_img = cv2.GaussianBlur(img, (3,3), 0)
    plt.imshow(new_img)
    plt.title("GaussianBlur")
    plt.show()
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(160, 160), interpolation = cv2.INTER_AREA)
    plt.imshow(new_img)
    plt.title("GaussianBlur + resize")
    plt.show()
    # scale to ?x?x3
    #new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    plt.imshow(new_img)
    plt.title("GaussianBlur + resize + YUV")
    plt.show()
    
    plt.imshow(new_img[65:-25,:])
    plt.title("GaussianBlur + resize + YUV + cropping")
    plt.show()
    
    return new_img

name1 = "C:/Users/kgasb/Desktop/windows-sim/windows_sim/data/IMG/center_2017_11_13_17_12_05_903.jpg"
image1 = cv2.imread(name1)
image2 = preprocess_image(image1)
image3 = image2 / 255. - .5
plt.imshow(image3[65:-25,:])
plt.title("GaussianBlur + resize + YUV + cropping + norm")
plt.show()