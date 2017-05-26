import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
net = MNISTConvNet()
print(net)

input = Variable(torch.randn(1, 1, 28, 28))
out = net(input)
print(input, out, out.size())

target = Variable(torch.LongTensor([3]))
loss_fn = nn.CrossEntropyLoss()
err = loss_fn(out, target)
print(err)

print(net.conv1.weight.grad)
print(net.conv1.weight.data)
print(net.conv1.weight)

err.backward()

print(net.conv1.weight.grad.size())
print(net.conv1.weight.grad.data)
print(net.conv1.weight.grad)
print(net.conv1.weight.data)
print(net.conv1.weight)

print(net.conv1.weight.data.norm())
print(net.conv1.weight.grad.data.norm())
 
def printnorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size: ', input[0].size())
    print('output size: ', output.data.size())
    print('output norm: ', output.data.norm()) 
    
net.conv2.register_forward_hook(printnorm)  

out = net(input)  

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class: ' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size: ', grad_input[0].size()) 
    print('grad_output size: ', grad_output[0].size()) 
    print('grad_input norm: ', grad_input[0].data.norm())
    
net.conv2.register_backward_hook(printgradnorm)

out = net(input)
err = loss_fn(out, target)
err.backward()

class RNN(nn.Module):
    
    def __init__(self, data_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        input_size = data_size + hidden_size  # Why?
        print('input_size: ')
        print(input_size)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        print('data: ')
        print(data)
        print('last_hidden: ')
        print(last_hidden)
        print('input: ')
        print(input)
        hidden = self.i2h(input)
        output = self.h2o(hidden)
        return hidden, output
    
rnn = RNN(20, 10, 5)
print(rnn)

loss_fn = nn.MSELoss()

batch_size = 10
TIMESTEPS = 5

batch = Variable(torch.randn(batch_size, 20))
hidden = Variable(torch.zeros(batch_size, 10))
target = Variable(torch.zeros(batch_size, 5))

loss = 0

for t in range(TIMESTEPS):
    hidden, output = rnn(batch, hidden)
    print('hidden: ')
    print(hidden)
    print('output: ')
    print(output)
    err = loss_fn(output, target)
    print('err: ')
    print(err)
    loss = loss + err
    print('loss: ')
    print(loss)
loss.backward()
