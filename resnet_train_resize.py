
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


#Dataset and dataloader for train,val and test data

 

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   

transform1 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   
     
transform2 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224,224)),
    torchvision.transforms.RandomHorizontalFlip(1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   

trainset1 = torchvision.datasets.ImageFolder(root='classification_data/train_data',transform=transform)
trainset2 = torchvision.datasets.ImageFolder('classification_data/train_data',transform=transform2)

trainset = torch.utils.data.ConcatDataset([trainset1, trainset2])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4)

devset = torchvision.datasets.ImageFolder(root='classification_data/val_data',transform=transform)
devloader = torch.utils.data.DataLoader(devset, batch_size=256,
                                          shuffle=False, num_workers=4)


testset = torchvision.datasets.ImageFolder(root='classification_data/test_data',transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=4)





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




#Training Functions 


def train(model, data_loader, test_loader, task='Classification'): 
    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0
        accuracy = 0.0
        total = 0.0
        val = 0.0

        for batch_num, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            accuracy += torch.sum((torch.argmax(outputs,dim=1)==labels)).item()
            total += len(labels)

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0    
                
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
            del outputs

        
        print("Accuracy in train", (accuracy/total)*100.0)
        if task == 'Classification':
            train_loss, train_acc = test_classify(model, data_loader)
            val_loss, val_acc = test_classify(model, test_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc*100.0, val_loss, val_acc*100.0))

            
            if val_loss >= val and epoch%5==0:
                temp_path = "res_out_" + str(epoch+1) + ".pth"
                torch.save(model,temp_path)
            val = val_loss
            
            scheduler.step()



def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1].detach()
        
        loss = criterion(outputs, labels)
        accuracy += torch.sum((torch.argmax(outputs,dim=1)==labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])

        del feats
        del labels
        del outputs

    model.train()
    return np.mean(test_loss), accuracy/total


#Putting in all the values 


numEpochs = 40
num_feats = 3

learningRate = 1e-3
weightDecay = 5e-5

hidden_sizes = [64, 128, 256, 512]
num_classes = len(trainset1.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = Network(num_feats, hidden_sizes, num_classes)
network.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(network.parameters(), lr = 1e-3, weight_decay=weightDecay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.85)   

network.train()
network.to(device)
train(network, trainloader, testloader)










