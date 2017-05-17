from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.randn(5, 3)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(x)
print(y)
print(x + y)

z = torch.add(x, y)
print(z)

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
out = torch.add(x, y, out=result)
print(result)
print(out)

y.add_(x)
print(y)

print(x)
print(x[:, 1])
print(x[2:3])

a = torch.ones(5, 5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = x + y
    print(x)
    print(y)
    print(z)
else:
    print(torch.cuda.is_available())