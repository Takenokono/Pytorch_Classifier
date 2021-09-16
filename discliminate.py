import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import cv2
from tqdm import tqdm


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# class ImageTransfrom():
#     def __init__(self,resize, mean, std):
#         self.data_transform ={
#             'train' : transforms.Compose([
#                 #transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)                
#             ]),
#             'val' : transforms.Compose([
#                 transforms.Resize(resize),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean, std)
#             ])
#         }
    
#     def __call__(self,img,phase='train'):
#         return self.data_transform[phase](img)


def transform_data(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor()
            #transforms.Normalize(MEAN,STD)
        ]
        )
    return transform

image_size = 224
batch_size = 12
root = "./dogs"

#Images Load from `./dogs` using torchvision.datasets.ImageFolder
dataset = ImageFolder(root, transform_data(image_size))
print(len(dataset))

n_samples = len(dataset) 
train_size = n_samples * 0.8 

subset1_indices = list(range(0,int(train_size))) 
subset2_indices = list(range(int(train_size),n_samples)) 

train_dataset = torch.utils.data.Subset(dataset, subset1_indices)
val_dataset   = torch.utils.data.Subset(dataset, subset2_indices)
print("train data : ",len(train_dataset))
print("val data : ", len(val_dataset))

# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False)
dataloaders_dict = {"train": train_dataloader, "val":val_dataloader}

batch_iterator = iter(dataloaders_dict["train"]) # Check
inputs , labels = next(batch_iterator)
print("inputs size : ",inputs.size())
print("labels : ", labels)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




ndf = 64
nc = 3


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Layer1
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer2
            nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Layer3
            nn.Conv2d(ndf, ndf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer4
            nn.Conv2d(ndf*2, ndf*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Layer5
            nn.Conv2d(ndf*2, ndf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer6
            nn.Conv2d(ndf*4, ndf*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Layer7
            nn.Conv2d(ndf*4, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer8
            nn.Conv2d(ndf*8, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False),
            # Layer9
            nn.Conv2d(ndf*8, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer10
            nn.Conv2d(ndf*8, ndf*8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False),
            )
        self.fc = nn.Sequential(
        # FC Layer1
        nn.Linear(25088, 4096, bias=True, device=None, dtype=None),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.5),
        # FC Layer2
        nn.Linear(4096, 100, bias=True, device=None, dtype=None),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.5),
        # FC Layer3
        nn.Linear(100, 3, bias=True, device=None, dtype=None),
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size()[0], -1) # POINT!!!!!
        x = self.fc(x)
        return x

net = Discriminator(device)
print(net)

# difine Loss
criterion = nn.CrossEntropyLoss()

# difine Optimizer
params = []
for name , param in net.named_parameters():
    params.append(param)
optimizer = optim.SGD(params=params,lr=0.001,momentum=0.9)

def train_model(net, dataloaders_dict,criterion, optimizer,num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        for phase in ['train','val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch== 0) and (phase=='train'):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{}Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

num_epochs = 2
train_model(net, dataloaders_dict,criterion, optimizer,num_epochs)