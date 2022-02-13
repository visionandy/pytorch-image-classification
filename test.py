import argparse
import os
import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--version", required=True, help="version number, v2, v3, etc?")
ap.add_argument("--task", required=True, help="which task? should be 1, 2, or 3")
args= vars(ap.parse_args())

# Paths for image directory and model

class_dict = {"task1":["M", "P"], "task2":["E", "G", "F"], "task3":["RED", "GREEN", "GRAY", "BLUE", "WHITE", "ORANGE", "YELLOW", "BLACK"]}


home_directory='/home/andywang/project/dataset/rock/'
version_num= args["version"]
task_v= "task" + args["task"]

# Set the train and validation directory paths
train_directory = home_directory+version_num+'/'+task_v+'/train'
valid_directory = home_directory+version_num+'/'+task_v+'/val'
# Set the model save path
SAVE_PATH="/home/andywang/project/model/"+version_num+'_'+task_v
MODEL=SAVE_PATH+"/model_epoch_50.pth" 


IMDIR='/home/andywang/project/dataset/Testing/combi/'#sys.argv[1])
out_path = '/home/andywang/project/dataset/inference'
if not os.path.exists(out_path):
    os.mkdir(out_path)
else:
    shutil.rmtree(out_path)
    os.mkdir(out_path)
#IMDIR = '/home/andywang/DB/800F_Grouped_by_Product_Family/800F__Push__Buttons__Illuminated/F'
#MODEL='models/resnet18.pth'

task_path = os.path.join(out_path, task_v)
if not os.path.exists(task_path):
    os.mkdir(task_path)

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names = class_dict[task_v]

# Retreive 9 random images from directory
files=Path(IMDIR).resolve().glob('*.*')
images=random.sample(list(files), 9)

# Configure plots
fig = plt.figure(figsize=(9,9))
rows,cols = 3,3

# Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Perform prediction and plot results
with torch.no_grad():
    for num,img in enumerate(images):
        img_fpath = img
        img_file_name = os.path.basename(img)
        img=Image.open(img).convert('RGB')
        inputs=preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)    
        label=class_names[preds]
        label_path = os.path.join(task_path, label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        shutil.copyfile(img_fpath, os.path.join(label_path, img_file_name))

        plt.subplot(rows,cols,num+1)
        plt.title("Pred: " + label + ' ' + img_file_name)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
'''
Sample run: python test.py test
'''
