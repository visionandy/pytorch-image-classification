import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths for image directory and model



home_directory='/home/andywang/project/dataset/rock/'
version_num='v2'
task_v='task2'
# Set the train and validation directory paths
train_directory = home_directory+version_num+'/'+task_v+'/train'
valid_directory = home_directory+version_num+'/'+task_v+'/val'
# Set the model save path
SAVE_PATH="/home/andywang/project/model/"+version_num+'_'+task_v
MODEL=SAVE_PATH+"/model_epoch_50.pth" 


IMDIR='/home/andywang/project/dataset/Testing/combi/'#sys.argv[1])
#MODEL='models/resnet18.pth'

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names=['part1','part2','part3']

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
         img=Image.open(img).convert('RGB')
         inputs=preprocess(img).unsqueeze(0).to(device)
         outputs = model(inputs)
         _, preds = torch.max(outputs, 1)    
         label=class_names[preds]
         plt.subplot(rows,cols,num+1)
         plt.title("Pred: "+label)
         plt.axis('off')
         plt.imshow(img)
    plt.show()
'''
Sample run: python test.py test
'''
