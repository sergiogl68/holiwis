import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import os.path as osp
from tqdm import tqdm
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)

#trainloader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([trainset,testset], batch_size=50, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(128*5*5, 10)
    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.max_pool2d(x,2)
        x = F.dropout(F.selu(self.conv3(x)),0.25)
        x = F.dropout(F.selu(self.conv4(x)),0.25)
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*5*5)
        x = self.fc1(x)
        return x


net=Net()
net=net.cuda()

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

for epoch in range(6):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        # get the inputs
        inputs, labels = data
        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.data[0]
        if i % 200 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
net.eval()
for data in testloader:
    images, labels = data
    outputs = net(Variable(images).cuda())
    outputs = outputs.cpu()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

labels = {} 
files = glob.glob("images/*.jpg")
for file in files:
    _id, _ = osp.splitext(osp.basename(file))
    img = Image.open(file)
    img = transform(img)
    if img.size(0) == 1:
        img = torch.stack([img] * 3, dim=1).squeeze()
    img = Variable(img, volatile=True).unsqueeze(0).cuda()
    print(img.size())
    output = net(img)
    output = output.cpu()
    _, predicted = torch.max(output.data, 1)
    labels[_id] = predicted[0]

with open('pred4p.txt', 'w') as f:
    f.write('\n'.join(['{0},{1}'.format(k, labels[k]) for k in labels]))

