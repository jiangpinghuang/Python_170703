import numpy as np
import torch

torch.manual_seed(1234)
    
# load Embedding
def loadEmbedFile(embFile):
    input = open(embFile, 'r')
    lines = []
    for line in input:
        lines.append(line)
    num = len(lines)
    dim = len(lines[0].split(' ')) - 1
    
    voc = []
    emb = torch.FloatTensor(num, dim)
    
    for i in range(num):
        token = lines[i].split(' ')
        voc.append(token[0])
        for j in range(1, dim + 1):
            emb[i][j-1] = float(token[j])
    
    return voc, emb

def buildVocab(line):
    word = {}
    index = {}
    size = 0
    
    tokens = line.split(' ')
    for i in range(len(tokens)):
        if tokens[i] not in word:
            word[size] = tokens[i]
            size += 1
    

def readMSCFile(trainFile, testFile):
    word_to_index = {}
    index_to_word = {}
    total_words = 0
    with open(trainFile, 'r') as train:
        for line in train:
            s = line.split('\t')  
            tokens = s[2].split(' ')
                      
        
    with open(testFile, 'r') as test:
        for line in test:
            s = line.split('\t')
            
def compMatrix(line):
    n_row = 0
    tokens = line.split(' ')
    
    for i in range(len(tokens)):
        if tokens[i][0:1] == 'B' or tokens[i][0:1] == 'O':
            n_row += 1
        else:
            continue
    return n_row
        

embFile = "/home/hjp/Workspace/Workshop/Corpus/bin/text.txt"
voc, emb = loadEmbedFile(embFile)  

trainFile = "/home/hjp/Workspace/Workshop/Corpus/msc/train.txt"
testFile = "/home/hjp/Workspace/Workshop/Corpus/msc/test.txt"

readMSCFile(trainFile, testFile)

