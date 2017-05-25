
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

def build_chunk_vec(sent, line, dim, mcl):
    num = 0
    tag = line.split(' ')
    for i in range(len(tag)):
        # print(tag[i])
        if tag[i][0:1] == 'I':
            num += 1
            
    chunk = torch.FloatTensor(mcl, dim).zero_()
    
    id = 0    
    
    for k in range(len(tag)):
        if tag[k][0:1] == 'B':
            if k == 0:
                chunk[id:id + 1] = sent[k:k + 1]
            else:
                id += 1
                chunk[id:id + 1] = sent[k:k + 1]
        elif tag[k][0:1] == 'I':
            chunk[id:id + 1] = chunk[id:id + 1].add_(sent[k:k + 1])
        else:
            if k == 0:
                chunk[id:id + 1] = sent[k:k + 1]
            else:
                id += 1
                chunk[id:id + 1] = sent[k:k + 1]

    return chunk          


def build_sent_vec(filepath, embed, dim, sn, msl, mcl):
    oov = 0
    idx = 0
    lsm = torch.FloatTensor(sn, msl, dim).zero_()
    rsm = torch.FloatTensor(sn, msl, dim).zero_()
    lcm = torch.FloatTensor(sn, mcl, dim).zero_()
    rcm = torch.FloatTensor(sn, mcl, dim).zero_()
    lbl = torch.IntTensor(sn).zero_()
    with open(filepath, 'r') as f:
        for line in f:
            print('idx: ', idx)
            sents = line.split('\t')
            lbl[idx:idx+1] = int(sents[0])
            lsent = sents[2].lower().split(' ')
            rsent = sents[6].lower().split(' ')
            # lsentm = torch.FloatTensor(len(lsent), dim)
            # rsentm = torch.FloatTensor(len(rsent), dim)
            lsentm = torch.FloatTensor(msl, dim).zero_()
            rsentm = torch.FloatTensor(msl, dim).zero_()            
            for i in range(len(lsent)):
                if lsent[i] in embed:
                    lsentm[i:i + 1] = torch.from_numpy(embed.get(lsent[i]))
                else:
                    oov += 1
                    lsentm[i:i + 1] = torch.FloatTensor(1, dim).normal_(0, 0.1)
            lchunk = build_chunk_vec(lsentm, sents[4], dim, mcl)
            
            for i in range(len(rsent)):
                if rsent[i] in embed:
                    rsentm[i:i + 1] = torch.from_numpy(embed.get(rsent[i]))
                else:
                    oov += 1
                    rsentm[i:i + 1] = torch.FloatTensor(1, dim).normal_(0, 0.1)
            rchunk = build_chunk_vec(rsentm, sents[8], dim, mcl)
            
            # print(lsentm)
            # print(lchunk)
            # print(rsentm)
            # print(rchunk)
            lsm[idx:idx + 1, :, :] = lsentm
            rsm[idx:idx + 1, :, :] = rsentm
            lcm[idx:idx + 1, :, :] = lchunk
            rcm[idx:idx + 1, :, :] = rchunk
            idx += 1
    return lsm, rsm, lcm, rcm, lbl
            
            
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
    for i in range(50):
        ltrsent, rtrsent, ltrchunk, rtrchunk, trlabel = build_sent_vec(trainfile, embed, dim, trsn, msl, mcl)
        ltesent, rtesent, ltechunk, rtechunk, telabel = build_sent_vec(testfile, embed, dim, trsn, msl, mcl)

    print(trlabel[0:10])
    print(telabel[100:110])
    print('finished!')

if __name__ == '__main__':
    main()

