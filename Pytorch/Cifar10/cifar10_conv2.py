import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import random

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), #data augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def augment_data(image):
    value = random.randint(0,180)
    if value > 140:
        if random.random() < 0.6:
            value = value / 2.0
    op = random.random()
    if op < 0.5:
        image = image * (1 - value/1000.0)
    else:
        image = image * (1 + value/1000.0)
    return image



class Conv2Net(nn.Module):
    def __init__(self):
        super(Conv2Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.bn1 = nn.BatchNorm2d(64)


        self.conv2_1 = nn.Conv2d(64,64,3)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64,64,3)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv2_3 = nn.Conv2d(128,128,3)
        self.bn2_3 = nn.BatchNorm2d(128)

        #pool

        self.conv3_1 = nn.Conv2d(128,128,3)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128,128,3)
        self.bn3_2 = nn.BatchNorm2d(128)


        self.conv3_3 = nn.Conv2d(256,256,3)
        self.bn3_3 = nn.BatchNorm2d(256)


        self.conv4_1 = nn.Conv2d(256,256,3)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256,256,3)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        self.conv4_3 = nn.Conv2d(512,512,3)
        self.bn4_3 = nn.BatchNorm2d(512)


        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(512 * 5 * 5, 1024)
        self.bnf1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.bnf2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 1024)
        self.bnf3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, 10)
        
    def forward(self, x):
        #x = augment_data(x)

        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.dropout(x,0.25)

        out1 = F.leaky_relu(self.conv2_1(x))
        out1 = self.bn2_1(out1)
        out1 = F.dropout(out1, 0.25)

        out2 = F.leaky_relu(self.conv2_2(x))
        out2 = self.bn2_2(out2)
        out2 = F.dropout(out2, 0.25)

        x = torch.cat((out1, out2), 1)

        x = F.leaky_relu(self.conv2_3(x))
        x = self.bn2_3(x)
        x = F.dropout(x,0.25)


        x = self.pool(x)

        out1 = F.leaky_relu(self.conv3_1(x))
        out1 = self.bn3_1(out1)
        out1 = F.dropout(out1, 0.25)

        out2 = F.leaky_relu(self.conv3_2(x))
        out2 = self.bn3_2(out2)
        out2 = F.dropout(out2, 0.25)

        x = torch.cat((out1, out2), 1)

        x = F.leaky_relu(self.conv3_3(x))
        x = self.bn3_3(x)
        x = F.dropout(x, 0.25)


        out1 = F.leaky_relu(self.conv4_1(x))
        out1 = self.bn4_1(out1)
        out1 = F.dropout(out1, 0.25)

        out2 = F.leaky_relu(self.conv4_2(x))
        out2 = self.bn4_2(out2)
        out2 = F.dropout(out2, 0.25)

        x = torch.cat((out1, out2), 1)

        x = F.leaky_relu(self.conv4_3(x))
        x = self.bn4_3(x)
        x = F.dropout(x, 0.25)

        #print x.size()

        x = x.view(-1, 512 * 5 * 5)

        x = F.leaky_relu(self.fc1(x))
        x = self.bnf1(x)
        x = F.dropout(x,0.3)

        x = F.leaky_relu(self.fc2(x))
        x = self.bnf2(x)
        x = F.dropout(x,0.3)

        x = F.leaky_relu(self.fc3(x))
        x = self.bnf3(x)
        x = F.dropout(x,0.3)

        x = self.fc4(x)

        return x




model = Conv2Net()



model.cuda()

print(str(model))


criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001/100)

for epoch in range(50):
    loss_tmp = 0.0
    #changing learning rate
    if epoch == 5:
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001/100)
    if epoch == 10:
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001/100)
    if epoch == 15:
        optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.00001/100) 
    if epoch == 20:
        optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00001/100)
    if epoch == 25:
        optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.00001/100)
    if epoch == 30:
        optimizer = optim.Adam(model.parameters(), lr=0.000000001, weight_decay=0.00001/100)
    if epoch == 40:
        optimizer = optim.Adam(model.parameters(), lr=0.0000000001, weight_decay=0.00001/100)     
    for i, (inputs,labels) in enumerate(trainloader, 0):
        inputs,labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_tmp += loss.item()
        if i % 250 == 249:
            print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, loss_tmp / 250))
            loss_tmp = 0.0
        if i == 249:
            correct = 0
            total = 0
            with torch.no_grad():
                for (images, labels) in testloader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the model on 10000 test images: %.4f %%' % (
                (100.0 * correct) / total))

print('Training is finished.')


correct = 0
total = 0
with torch.no_grad():
    for (images, labels) in testloader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on 10000 test images: %.4f %%' % (
    (100.0 * correct) / total))


