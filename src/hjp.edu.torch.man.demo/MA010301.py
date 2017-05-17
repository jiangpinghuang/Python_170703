import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3)
        print('self.conv1: ')
        print(self.conv1)
        #self.conv2 = nn.Conv2d(2, 4, 5)
        self.fc1 = nn.Linear(1 * 3 * 3, 5)
        self.fc2 = nn.Linear(5, 2)
        #self.fc3 = nn.Linear(10, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[0:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

params = list(net.parameters())
print(len(params))

def num_param_model(nnm):
    params = list(nnm.parameters())
    num_params = 0
    for i in range(len(params)):
        single_layer = 1
        size = params[i].size()[0:]
        for j in size:
            single_layer *= j
        num_params += single_layer
    return num_params

print('number of parameters: ')
print(num_param_model(net))
        
print(params[0].size())
print(params[0])
# print(params[1].size())
# print(params[1])
# print(params[2].size())
# print(params[2])
# print(params[3].size())
# print(params[3])
# print(params[4].size())
# print(params[4])
# print(params[5].size())
# print(params[5])
# print(params[6].size())
# print(params[6])
# print(params[7].size())
# print(params[7])
# print(params[8].size())
# print(params[8])
# print(params[9].size())
# print(params[9])

input = Variable(torch.randn(1, 1, 8, 8))
print(input)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 2))

output = net(input)
target = Variable(torch.range(1, 2))
print(target)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.creator)
print(loss.creator.previous_functions[0][0])
print(loss.creator.previous_functions[0][0].previous_functions[0][0])
print(loss.creator.previous_functions[0][0].previous_functions[0][0].previous_functions[0][0])

net.zero_grad()

print('before backward: ')
print(net.conv1.bias)
print(net.conv1.bias.data)
print(net.conv1.bias.grad)
print(net.conv1.bias.grad.data)
# print(net.conv1.weight)
# print(net.conv1.weight.data)
# print(net.conv1.weight.grad)
# print(net.fc1.bias)
# print(net.fc1.bias.data)
# print(net.fc1.bias.grad)
# print(net.fc1.weight)
# print(net.fc1.weight.data)
# print(net.fc1.weight.grad)
# print(net.fc2.bias)
# print(net.fc2.bias.data)
# print(net.fc2.bias.grad)
# print(net.fc2.weight)
# print(net.fc2.weight.data)
# print(net.fc2.weight.grad)

loss.backward()

print('after backward: ')
print(net.conv1.bias)
print(net.conv1.bias.data)
print(net.conv1.bias.grad)
print(net.conv1.bias.grad.data)
# print(net.conv1.weight)
# print(net.conv1.weight.data)
# print(net.conv1.weight.grad)
# print(net.fc1.bias)
# print(net.fc1.bias.data)
# print(net.fc1.bias.grad)
# print(net.fc1.weight)
# print(net.fc1.weight.data)
# print(net.fc1.weight.grad)
# print(net.fc2.bias)
# print(net.fc2.bias.data)
# print(net.fc2.bias.grad)
# print(net.fc2.weight)
# print(net.fc2.weight.data)
# print(net.fc2.weight.grad)

learning_rate = 1e-2

print("before: ")
for f in net.parameters():
    print(f.data)
    print(f.grad.data)

for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

print("after: ")    
for f in net.parameters():
    print(f.data)
    print(f.grad.data)
    
import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
print("input: ")
print(input)
print("output: ")
print(output)
print("target: ")
print(target)
loss = criterion(output, target)
print("loss: ")
print(loss)
print("weight before step(): ")
for f in net.parameters():
    print(f.data)
print("gradient before step(): ")
for f in net.parameters():
    print(f.grad.data)
loss.backward()
optimizer.step()
print("weight after step(): ")
for f in net.parameters():
    print(f.data)
print("gradient after step(): ")
for f in net.parameters():
    print(f.grad.data)