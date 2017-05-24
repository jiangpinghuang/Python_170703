
import torch

import numpy as np

torch.manual_seed(1234567890)


def load_embed_file(embfile):
    embed = {}
    with open(embfile, 'r') as f:
        for line in f:
            elem = line.split()
            word = elem[0]
            value = np.asarray(elem[1:], dtype='float32')
            embed[word] = value
        f.close()
    print('Found %s word vectors.' % len(embed))
    return embed
  
    
def read_msc_file(trainfile, testfile):
    word_to_index = {}
    index_to_word = {}
    index = 0
    count = 0
    MAX_SENT_LENGTH = 0
    MAX_CHUNK_LENGTH = 0
    TRAIN_SENT_NUM = 0
    TEST_SENT_NUM = 0
    
    with open(trainfile, 'r') as train:
        for line in train:
            TRAIN_SENT_NUM += 1
            sents = line.split('\t')
            token = sents[2].split(' ')
            if MAX_SENT_LENGTH < len(token):
                MAX_SENT_LENGTH = len(token)
            for i in range(len(token)):
                if token[i] not in word_to_index:
                    word_to_index[token[i]] = index
                    index_to_word[index] = token[i]
                    index += 1
                    
            tag = sents[4].split(' ')            
            for i in range(len(tag)):
                if tag[i][0:1] == 'B' or tag[i][0:1] == 'O':
                    count += 1
            if MAX_CHUNK_LENGTH < count:
                MAX_CHUNK_LENGTH = count
                count = 0
            else:
                count = 0
                    
            token = sents[6].split(' ')
            if MAX_SENT_LENGTH < len(token):
                MAX_SENT_LENGTH = len(token)
            for i in range(len(token)):
                if token[i] not in word_to_index:
                    word_to_index[token[i]] = index
                    index_to_word[index] = token[i] 
                    index += 1
            
            tag = sents[8].split(' ')            
            for i in range(len(tag)):
                if tag[i][0:1] == 'B' or tag[i][0:1] == 'O':
                    count += 1
            if MAX_CHUNK_LENGTH < count:
                MAX_CHUNK_LENGTH = count
                count = 0
            else:
                count = 0
           
    with open(testfile, 'r') as test:
        for line in test:
            TEST_SENT_NUM += 1
            sents = line.split('\t')
            token = sents[2].split(' ')
            if MAX_SENT_LENGTH < len(token):
                MAX_SENT_LENGTH = len(token)
            for i in range(len(token)):
                if token[i] not in word_to_index:
                    word_to_index[token[i]] = index
                    index_to_word[index] = token[i]
                    index += 1
                    
            tag = sents[4].split(' ')            
            for i in range(len(tag)):
                if tag[i][0:1] == 'B' or tag[i][0:1] == 'O':
                    count += 1
            if MAX_CHUNK_LENGTH < count:
                MAX_CHUNK_LENGTH = count
                count = 0
            else:
                count = 0
                    
            token = sents[6].split(' ')
            if MAX_SENT_LENGTH < len(token):
                MAX_SENT_LENGTH = len(token)
            for i in range(len(token)):
                if token[i] not in word_to_index:
                    word_to_index[token[i]] = index
                    index_to_word[index] = token[i] 
                    index += 1
                    
            tag = sents[8].split(' ')            
            for i in range(len(tag)):
                if tag[i][0:1] == 'B' or tag[i][0:1] == 'O':
                    count += 1
            if MAX_CHUNK_LENGTH < count:
                MAX_CHUNK_LENGTH = count
                count = 0
            else:
                count = 0
                    
    return word_to_index, index_to_word, MAX_SENT_LENGTH, MAX_CHUNK_LENGTH, TRAIN_SENT_NUM, TEST_SENT_NUM

def build_chunk_vec(sent, line, dim):
    num = 0
    tag = line.split(' ')
    for i in range(len(tag)):
        print(tag[i])
        if tag[i][0:1] == 'I':
            num += 1
            
    chunk = torch.FloatTensor(len(tag) - num, dim)
    
    id = 0    
    
    for k in range(len(tag)):
        if tag[k][0:1] == 'B':
            if k == 0:
                chunk[id:id+1] = sent[k:k+1]
            else:
                id += 1
                chunk[id:id+1] = sent[k:k+1]
        elif tag[k][0:1] == 'I':
            chunk[id:id+1] = chunk[id:id+1].mul_(sent[k:k+1])
        else:
            if k == 0:
                chunk[id:id+1] = sent[k:k+1]
            else:
                id += 1
                chunk[id:id+1] = sent[k:k+1]

    return chunk          


def build_sent_vec(filepath, embed, dim):
    oov = 0
    with open(filepath, 'r') as f:
        for line in f:
            sents = line.split('\t')
            lsent = sents[2].lower().split(' ')
            rsent = sents[6].lower().split(' ')
            lsentm = torch.FloatTensor(len(lsent), dim)
            rsentm = torch.FloatTensor(len(rsent), dim)
            
            for i in range(len(lsent)):
                if lsent[i] in embed:
                    lsentm[i:i+1] = torch.from_numpy(embed.get(lsent[i]))
                else:
                    oov += 1
                    lsentm[i:i+1] = torch.FloatTensor(1, dim).normal_(0, 0.1)
            lchunk = build_chunk_vec(lsentm, sents[4], dim)
            
            for i in range(len(rsent)):
                if rsent[i] in embed:
                    rsentm[i:i+1] = torch.from_numpy(embed.get(rsent[i]))
                else:
                    oov += 1
                    rsentm[i:i+1] = torch.FloatTensor(1, dim).normal_(0, 0.1)
            rchunk = build_chunk_vec(rsentm, sents[8], dim)
            
            print(lsentm)
            print(lchunk)
            print(rsentm)
            print(rchunk)
            
            
def main():
    
    word_to_index = {}
    index_to_word = {}
    glovefile = '/home/hjp/Downloads/glove.txt'
    embfile = '/home/hjp/Workspace/Workshop/Corpus/bin/text.txt'
    trainfile = "/home/hjp/Workspace/Workshop/Corpus/msc/train.txt"
    testfile = "/home/hjp/Workspace/Workshop/Corpus/msc/test.txt"
    demofile = '/home/hjp/Downloads/demo.txt'
    dim = 20
    
    embed = load_embed_file(embfile)
    
    word_to_index, index_to_word, msl, mcl, trsn, tesn = read_msc_file(trainfile, testfile)

    build_sent_vec(demofile, embed, dim)
    
    print('msl: ', msl)
    print('mcl: ', mcl)
    print('trsn: ', trsn)
    print('tesn: ', tesn)

if __name__ == '__main__':
    main()

