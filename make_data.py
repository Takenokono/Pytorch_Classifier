from glob import glob
import xml.etree.ElementTree as ET
from pprint import pprint
import os 
import cv2

img_dir = './all-dogs'
anno_dir_Chihuahua = './Annotation/n02085620-Chihuahua'
anno_dir_Japanese_spaniel = './Annotation/n02085782-Japanese_spaniel'
anno_dir_Maltese_dog ='./Annotation/n02085936-Maltese_dog'

# all image path
img_list = glob(img_dir+'/*')

# retrive infomation from Annotation Data
anno_lists = [anno_dir_Chihuahua,anno_dir_Japanese_spaniel,anno_dir_Maltese_dog]

flag = 0
for anno_list in anno_lists:
    anno_files = glob(anno_list+'/*')
    #print(anno_files)
    dog_name = []
    file_name = []
    for anno_file in anno_files:
        tree = ET.parse(anno_file)
        root = tree.getroot()
        
        for name in root.iter('filename'):
                file_name.append(name.text)
                break
    
    for f in file_name:
        f = f + '.jpg'
        for img_path in img_list:
            #print(os.path.basename(img_path))
            if os.path.basename(img_path) == f:
                i = cv2.imread(img_path)
                
                if flag==0:
                    cv2.imwrite('./dogs/Chihuahua/'+f,i)
                if flag==1:
                    cv2.imwrite('./dogs/Japanese_spaniel/'+f,i)
                if flag==2:
                    cv2.imwrite('./dogs/Maltese_dog/'+f,i)
    flag = flag + 1







