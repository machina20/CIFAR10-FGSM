import torch
import torch.nn as nn  # object oriented programming
import torch.nn.functional as F  # functions
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
import argparse
import numpy as np

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', help="Learning rate", default='0.001', type=float)
    parser.add_argument('-e', '--epochs', help="Number of Epochs", default='10', type=int)
    parser.add_argument('-b', '--batch_size', help="Batch size", default='4', type=int)
    parser.add_argument('-m', '--momentum', help="Momentum", default='0.9', type=float)

    args = parser.parse_args()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = args.batch_size  # 4
    EPOCHS = args.epochs  # 7
    lr = args.learning_rate  # 0.001
    momentum = args.momentum # 0.9

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    accuracy = str(100 * correct / total)

    cwd = os.getcwd()
    folder = "cifar_runs"
    path = os.path.join(cwd, folder)
    if not os.path.exists(path):
        os.mkdir(path)

    batch_size = str(batch_size)  # 4
    EPOCHS = str(EPOCHS)  # 7
    lr = str(lr)  # 0.001
    momentum = str(momentum)  # 0.9
    loss = str(loss)
    print(type(momentum)) # prints str
    print(type(loss)) #
    

    PATH = './cifar_runs/cifar_' + lr + '_' + EPOCHS + '.pth'
    torch.save(net.state_dict(), PATH)

    filename = "lr" + lr + "_epochs" + EPOCHS + ".txt"
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            f.write("Learning-rate = " + lr +
                    "\n\nEpochs = " + EPOCHS +
                    "\n\nLoss = " + loss +
                    "\n\nMomentum = " + momentum +
                    "\n\nAccuracy = " + accuracy)