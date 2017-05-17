import torch
from torch.autograd import Variable

x = Variable(torch.rand(2, 3), requires_grad=True)
print(x)
print(x.data)
print(x.grad)
print(x.creator)

y = x + 2
print(y)
print(y.data)
print(y.grad)
print(y.creator)

z = y * y * 3
print(z)
print(z.data)
print(z.grad)
print(z.creator)

out = z.mean()
print(out)
print(out.data)
print(out.grad)
print(out.creator)

print(z, out)

out.backward()
print(out.data)
print(out.grad)
print(out.creator)

print(x.data)
print(x.grad)
print(x.creator)

print(y.data)
print(y.grad)
print(y.creator)

print(x.data)
print(x.grad)

x = torch.randn(3)
x = Variable(x, requires_grad=True)
print(x)
print(x.data)
print(x.grad)
print(x.creator)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

gradients = torch.FloatTensor([0.0001, 0.001, 0.01, 0.1, 1.0, 1])
y.backward(gradients)
print(x.grad)

