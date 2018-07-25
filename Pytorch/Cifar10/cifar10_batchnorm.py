import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim



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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(256*2*2, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.bn1(x)
        x = F.dropout(x, 0.2)

        x = F.leaky_relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, 0.3)

        x = self.pool(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.bn3(x)
        x = F.dropout(x, 0.3)

        x = self.pool(x)
        x = self.pool(F.leaky_relu(self.conv4(x)))

        x = x.view(-1, 256*2*2)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn4(x)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x))
        x = self.bn5(x)
        x = F.dropout(x, 0.3)
        x = self.fc3(x)
        return x
    def name(self):
    	return "ConvNet"



model = ConvNet()

model = model.cuda()

print(str(model))


criterion = nn.CrossEntropyLoss()

#optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-5)


optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001/100)

for epoch in range(100):
    loss_tmp = 0.0
    #changing learning rate
    if epoch == 20:
        optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.0001/100)
    if epoch == 50:
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001/100)
    if epoch == 80:
        optimizer = optim.Adam(model.parameters(), lr=0.000005, weight_decay=0.00001/100)        
    for i, (inputs,labels) in enumerate(trainloader, 0):
        inputs,labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_tmp += loss.item()
        if i % 625 == 624:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss_tmp / 625))
            loss_tmp = 0.0
        if i == 3124:
            correct = 0
            total = 0
            with torch.no_grad():
                for (images, labels) in testloader:
                    images, labels = images.cuda(), labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the model on 10000 test images: %d %%' % (
                100 * correct / total))

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

print('Accuracy of the model on 10000 test images: %d %%' % (
    100 * correct / total))
