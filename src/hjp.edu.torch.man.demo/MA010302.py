import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.fc1 = nn.Linear(1 * 3 * 3, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[0:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
net = Net()

input = Variable(torch.randn(1, 1, 8, 8))
print(input)
output = net(input)
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
target = Variable(torch.rand(2))
print(target)
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()
print('before backward: ')
print(net.conv1.bias)
print(net.conv1.bias.grad)
print(net.conv1.weight)
print(net.conv1.weight.grad)
optimizer.step()
print('after backward: ')
print(net.conv1.bias)
print(net.conv1.bias.grad)
print(net.conv1.weight)
print(net.conv1.weight.grad)
