

import os
import numpy as np
from PIL import Image
from torch.utils import data

import pandas as pd
import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Building Blocks for RESNET 18

class BasicBlock(nn.Module):
    #Only for idx > 0
    def __init__(self,channel_size1, channel_size2, stride1=1, stride2=2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size1, channel_size2, kernel_size=3, stride=stride2, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(channel_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(channel_size2, channel_size2, kernel_size=3, stride=stride1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_size2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcut = nn.Conv2d(channel_size1, channel_size2, kernel_size=1, stride=stride2, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


''' RESNET 18 '''
class BasicBlock2(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        

        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            if idx==0:
              self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], 
                                            out_channels=self.hidden_sizes[idx+1], 
                                            kernel_size=7, stride=2, padding = 3, bias=False))
              self.layers.append(nn.BatchNorm2d(num_features=self.hidden_sizes[idx+1],eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
              self.layers.append(nn.ReLU(inplace=True))
              
              self.layers.append(nn.MaxPool2d(3, stride=2, padding=1, dilation=1, ceil_mode=False))
              # 2 times
              self.layers.append(nn.Sequential(BasicBlock2(channel_size = channel_size),
                                                BasicBlock2(channel_size = channel_size)))
            else:
              self.layers.append(BasicBlock(channel_size1=self.hidden_sizes[idx], channel_size2=self.hidden_sizes[idx+1]))
              self.layers.append(BasicBlock2(channel_size=channel_size))

        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.layers = nn.Sequential(*self.layers)
        
        self.linear1 = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)
        

        # For creating the embedding to be passed into the Center Loss criterion
        # self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        # self.relu_closs = nn.ReLU(inplace=True)
        
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
        
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        label_output = self.linear1(output)

        return output,label_output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)




ver_model = torch.load("res_out_26.pth")


#Custom Datset loader for the verification images 

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list=None):
        self.file_list = file_list
        self.normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if target_list:
            self.target_list = target_list
            self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path1,path2=self.file_list[index]
        #validation_path = ''
        img1=Image.open(path1)
        img2=Image.open(path2)
        img1= torchvision.transforms.Compose([
	 	torchvision.transforms.Resize(size = (224,224)),
     	torchvision.transforms.ToTensor(),
     	torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img1)
        img2= torchvision.transforms.Compose([
	 	torchvision.transforms.Resize(size = (224,224)),
     	torchvision.transforms.ToTensor(),
     	torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(img2)
        return img1,img2


cnt = 0

with open (r'verification_pairs_test.txt') as fp:
    line = fp.readline()
    data_lines=[]
    while line:
        #print("Line {}: {}".format(cnt, line.strip()))
        line = fp.readline()
        data_lines.append(line)
        cnt += 1
    data_lines = [x.strip() for x in data_lines] 
data_lines = [x.split() for x in data_lines]
data_lines=data_lines[:-1]

file_namelist=[]
target_list=[]
for i in data_lines:
    file_namelist.append(i[0:2])



f = open("./verification_pairs_test.txt",'r')
C=[]
i = 0
for line in f:
  if i==0:
    x, y = line.split(' ')
    # A.append(x)
    # B.append(y[:-1])
    C.append([x,y[:-1]])
    i = i+1


file_namelist.insert(0, C[0])
verset=ImageDataset(file_namelist[:],target_list[:])
verloader=torch.utils.data.DataLoader(verset, batch_size=1,shuffle=False,num_workers=4,drop_last = False)





output = []

for i,(img1,img2) in enumerate(verloader):
    img1 = img1.to(device)
    img2 = img2.to(device)
    out1 = ver_model(img1)[0].to('cpu')
    out2 = ver_model(img2)[0].to('cpu')
    cos_sim = F.cosine_similarity(out1,out2)
    val = cos_sim.detach().numpy()[0]
    output.append(val)

print(len(output))

#output.append(0.9)



f2 = open('submission.csv','w')
f2.write('Id,Category\n')
for i in range(len(output)):
    f2.write(file_namelist[i][0]+' '+file_namelist[i][1]+','+str(output[i])+'\n')
f2.close()

