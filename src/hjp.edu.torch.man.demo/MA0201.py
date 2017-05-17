import torch

a = torch.FloatTensor(5, 7)
print(a)

a = torch.randn(5, 7)
print(a)
print(a.size())

a.fill_(3.5)
print(a)

b = a.add(4.0)
print(b)

b = a[0, 3]
print(b)

b = a[:, 3:5]
print(b)

x = torch.ones(5, 5)
print(x)

z = torch.Tensor(5, 2)
z[:, 0] = 10
z[:, 1] = 100
print(z)

print(torch.LongTensor([4, 0]))
x.index_add_(1, torch.LongTensor([4, 0]), z)
print(x)

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    a = torch.LongTensor(10).fill_(3).cuda()
    print(type(a))
    b = a.cpu()
    
print(torch.cuda.is_available())
