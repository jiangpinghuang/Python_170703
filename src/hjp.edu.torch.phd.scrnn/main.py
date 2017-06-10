import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

nin = 100
din = 10
dim = 5
out = 2


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2*dim, dim)
        self.linear2 = nn.Linear(dim, out)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        p = Variable(torch.FloatTensor(torch.zeros(len(x[0]))))
        for i in range(len(x)-1):
            if i == 0:
                c1 = x[i]
                c2 = x[i+1]
                p = (torch.cat((c1, c2), 0)).view(1, -1)
                p = self.tanh(self.linear1(p))
            else:
                c1 = p.view(dim, -1)
                c2 = x[i+1]
                p = (torch.cat((c1, c2), 0)).view(1, -1)
                p = self.tanh(self.linear1(p))
        out = self.relu(self.linear2(p))
        out = self.softmax(out)
        return out


net = Net()
print net

params = list(net.parameters())
print len(params)

x = torch.FloatTensor(torch.randn(din, dim))
x = Variable(x)
print x

pred = net(x)
print "pred: "
print pred

target = Variable(torch.FloatTensor(torch.rand(2)).view(1, -1))
print "target:"
print target

criterion = nn.MSELoss()

learning_rate = 1e-2

optimizer = optim.SGD(net.parameters(), lr=learning_rate)

loss = criterion(pred, target)
print loss

optimizer.zero_grad()

loss.backward()

optimizer.step()
for i in range(300):
    x = torch.FloatTensor(torch.randn(din+i, dim))
    x = Variable(x)
    #print x

    pred = net(x)
    print "pred: "
    print pred

    loss = criterion(pred, target)
    print i, loss

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

x = torch.FloatTensor(torch.randn(din+25, dim))
x = Variable(x)
print x

pred = net(x)
print "pred: "
print pred
# x = torch.FloatTensor(torch.randn(din, dim))
# print x
# x = Variable(x)
# print x
# 
# p = torch.cat((x[0], x[1]), 0)
# print p
# 
# 
# net = Net()
# print net
# 
# 
# p = p.view(1, -1)
# print p
# 
# y = net(p)
# print y
