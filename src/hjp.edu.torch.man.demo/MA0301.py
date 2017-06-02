import numpy as np

N, D_in, H, D_out = 10, 20, 20, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

print('x: ')
print(x)
print('y: ')
print(y)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
print('w1: ')
print(w1)
print('w2: ')
print(w2)

learning_rate = 1e-3

for t in range(100):
    h = x.dot(w1)
    print('h: ')
    print(h)
    h_relu = np.maximum(h, 0)
    print('h_relu: ')
    print(h_relu)
    y_pred = h_relu.dot(w2)
    print('y_pred: ')
    print(y_pred)
    
    print('y_pred - y: ')
    print(y_pred)
    print(y)
    print(y_pred - y)
    
    loss = np.square(y_pred - y).sum()
    print('loss: ')
    print(t, loss)
    
    grad_y_pred = 2.0 * (y_pred - y)
    print('y_pred - y: ')
    print(y_pred - y)
    print('grad_y_pred: ')
    print(grad_y_pred)
    print('h_relu: ')
    print(h_relu)
    print(h_relu.T)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    print('grad_w2: ')
    print(grad_w2)
    print('w2.T: ')
    print(w2.T)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    print('grad_h_relu: ')
    print(grad_h_relu)
    print('grad_h: ')
    print(grad_h)
    grad_h[h < 0] = 0
    print('grad_h < 0: ')
    print(grad_h)
    print('x.T: ')
    print(x.T)
    grad_w1 = x.T.dot(grad_h)
    print('grad_w1: ')
    print(grad_w1)
    
    print('w1: ')
    print(w1)
    w1 -= learning_rate * grad_w1
    print('updated w1: ')
    print(w1)
    print('w2: ')
    print(w2)
    w2 -= learning_rate * grad_w2
    print('updated w2: ')
    print(w2)
