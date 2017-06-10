import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import RevNN

import data


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--emb_size', dest='embed_dim', type=int, help='embedding size', default=50)
parser.add_argument('--hid_size', dest='hiddn_dim', type=int, help='hidden size', default=300)
parser.add_argument('--seed',dest='seed',type=int, help='seed',default=1234567890)
parser.add_argument('--epochs', dest='epoch', type=int, help='epoch', default=1)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning rate', default=1e-3)
parser.add_argument('--num_class', dest='num_class', type=int, help='number of classes', default=2)
parser.add_argument('--emb_path', dest='embed_path', help='embedding path', default='/home/hjp/Downloads/msc/glove.6B.50d.txt')
parser.add_argument('--data_path', dest='data_path', help='data set path', default='/home/hjp/Downloads/msc/')
args = parser.parse_args()

torch.manual_seed(args.seed)

embed, vocab = data.build_embed(args.embed_path)
print torch.from_numpy(embed['world'])
word2idx, idx2word, trainlbl, testlbl = data.build_corpus(args.data_path)
print word2idx['chief']

trainlabel = torch.FloatTensor(torch.zeros(len(trainlbl), 1))
print trainlabel
for i in range(len(trainlbl)):
    print trainlbl[i]
    if trainlbl[i] == '1':
        trainlabel[i] = 1
    else:
        trainlabel[i] = 0
print trainlabel

testlabel = torch.FloatTensor(torch.zeros(len(testlbl), 1))
print testlabel
for i in range(len(testlbl)):
    print testlbl[i]
    if testlbl[i] == '1':
        testlabel[i] = 1
    else:
        testlabel[i] = 0
print testlabel

trainlabel = Variable(trainlabel)
testlabel = Variable(testlabel)

net = RevNN(args.embed_dim, args.hiddn_dim, args.num_class)
print net

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.learning_rate)

# target = Variable(torch.FloatTensor(torch.rand(args.epoch, 2)))
# print target


for i in range(args.epoch):
    index = 0
    with open('/home/hjp/Downloads/msc/train.txt', 'r') as f:
        for line in f:
            optimizer.zero_grad()
            lsentm, rsentm = data.build_vector(embed, args.embed_dim, vocab, line)
            pred = net(lsentm, rsentm, args.embed_dim)
            print torch.max(pred)
            print pred, trainlabel[index]
            #label = torch.IntTensor(len(trainlbl[index]))
            
            loss = criterion(pred, trainlabel[index])

            print index, loss
            loss.backward()
            optimizer.step() 
            index += 1       


for line in open('/home/hjp/Downloads/msc/test.txt', 'r'):
    lsentm, rsentm = data.build_vector(embed, args.embed_dim, vocab, line)
    pred = net(lsentm, rsentm, args.embed_dim)
    print pred



# for i in range(args.epoch):
#     optimizer.zero_grad()
#     x1 = Variable(torch.FloatTensor(torch.randn(i+50, args.embed_dim)))
#     x2 = Variable(torch.FloatTensor(torch.randn(2*i+50, args.embed_dim)))
#     pred = net(x1, x2, args.embed_dim)
#     loss = criterion(pred, target[i])
#     print i, loss
#     loss.backward()
#     optimizer.step()
#   
# x_1 = Variable(torch.FloatTensor(torch.randn(150, args.embed_dim)))
# x_2 = Variable(torch.FloatTensor(torch.randn(180, args.embed_dim)))
# pred = net(x_1, x_2, args.embed_dim)   
# print pred 

