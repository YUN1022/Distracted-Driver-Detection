import pandas as pd
import os, shutil
import cv2
import time

base_dir = 'D:\Final\code\LAB\keras\Distracted Driver Detection\imgs'
imgs = os.listdir(base_dir+"/test")

def job(data):
    temp = []
    for i in range(len(data)):
        img_name = base_dir+"/test/"+imgs[i]
        img = cv2.imread(img_name)
        img = img[:,120:-30]
        img = cv2.resize(img,(140,140))
        temp.append(img)
    return temp