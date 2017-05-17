import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)
print(x.data)
print(x.grad)
print(x.creator)

y = x + 2
print(y)
print(y.creator)

z = y * y * 3
out = z.mean()

print(z, out)

print(out)
print(out.creator)

print(x.grad)
out.backward()
print(x.grad)

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
print(x.grad)
y.backward(torch.ones(2, 2), retain_variables=True)
print(x.grad)

z = y * y 
print(z)
print(x.grad)
z.backward(torch.ones(2, 2), retain_variables=True)
print(x.grad)

gradient = torch.ones(2, 2)
y.backward(gradient)
print(y)

print(x.grad)

x = Variable(torch.ones(2, 2), requires_grad=True)
y = 2 * x + 2
print(x)
print(y)
gradient = torch.randn(2, 2)
print(gradient)
y.backward(gradient)
print(x.grad)
