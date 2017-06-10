import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init

def build_embed(path):
    embed = {}
    eword = []
    with open(path, 'r') as f:
        for line in f:
            token = line.split()
            word = token[0]
            eword.append(word)
            value = np.asarray(token[1:], dtype='float32')
            embed[word] = value
        f.close()
    print('Found %s word vectors.' % len(embed))
    return embed, eword    

def build_corpus(path):
    word2idx = {}
    idx2word = []
    trainlbl = []
    testlbl = []

    assert os.path.exists(path)
    
    train = os.path.join(path, 'train.txt')
    test = os.path.join(path, 'test.txt')
    
    with open(train, 'r') as f:
        for line in f:
            sents = line.split('\t')
            trainlbl.append(sents[0])
            words = sents[2].split()
            for word in words:
                if word not in word2idx:
                    idx2word.append(word)
                    word2idx[word] = len(idx2word) - 1
            words = sents[6].split()
            for word in words:
                if word not in word2idx:
                    idx2word.append(word)
                    word2idx[word] = len(idx2word) - 1
                    
    with open(test, 'r') as f:
        for line in f:
            sents = line.split('\t')
            testlbl.append(sents[0])
            words = sents[2].split()
            for word in words:
                if word not in word2idx:
                    idx2word.append(word)
                    word2idx[word] = len(idx2word) - 1
            words = sents[6].split()
            for word in words:
                if word not in word2idx:
                    idx2word.append(word)
                    word2idx[word] = len(idx2word) - 1
                    
    return word2idx, idx2word, trainlbl, testlbl

def build_vector(embed, dim, vocab, line):
    sents = line.split('\t')
    lsent = sents[2].split()
    rsent = sents[6].split()
    
    lsentm = Variable(torch.FloatTensor(torch.zeros(len(lsent), dim)))
    rsentm = Variable(torch.FloatTensor(torch.zeros(len(rsent), dim)))
    
    for i in range(len(lsent)):
        if lsent[i] in vocab:
            #print embed[lsent[i]]
            lsentm[i] = torch.from_numpy(embed[lsent[i]]).view(1, dim)
        else:
            w = torch.Tensor(1, dim)
            w = nn.init.normal(w, 0, 0.1)
            lsentm[i] = w
            
    for i in range(len(rsent)):
        if rsent[i] in vocab:
            #print embed[rsent[i]]
            rsentm[i] = torch.from_numpy(embed[rsent[i]]).view(1, dim)
        else:
            w = torch.Tensor(1, dim)
            w = nn.init.normal(w, 0, 0.1)
            rsentm[i] = w
            
    return lsentm, rsentm

# path = "/home/hjp/Downloads/msc/"
# word2idx, idx2word, trainlbl, testlbl = build_corpus(path)
# print word2idx
# print idx2word
# print len(word2idx)
# print len(idx2word)
# print trainlbl
# print len(trainlbl)
# print len(testlbl)
#def word2emb(word, emb):
    




# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {}
#         self.idx2word = []
#         
#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         return self.word2idx[word]
#     
#     def __len__(self):
#         return len(self.idx2word)
#         
#         
# 
# def vocab(x):
#     x.sum()
#     
# with open('/home/hjp/Downloads/msc/glove.6B.50d.txt') as f:
#     for line in f:
#         print line
#         
# def readMSC():
#     return 