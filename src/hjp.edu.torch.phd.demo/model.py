import torch
import torch.nn as nn
from torch.autograd import Variable


class RevNN(nn.Module):
    
    def __init__(self, word_dim, hidden_size, num_class):
        super(RevNN, self).__init__()
        self.linear1 = nn.Linear(2*word_dim, word_dim)        
        self.linear2 = nn.Linear(word_dim, hidden_size)        
        self.linear3 = nn.Linear(hidden_size, word_dim)
        self.linear4 = nn.Linear(2*word_dim, num_class)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, input2, word_dim):
        p = Variable(torch.FloatTensor(torch.zeros(word_dim)))        
        p2 = Variable(torch.FloatTensor(torch.zeros(word_dim)))
        
        for i in range(len(input)-1):
            if i == 0:
                p = self.tanh(self.linear3(self.tanh(self.linear2(self.relu(self.linear1((torch.cat((input[i], input[i+1]), 0)).view(1, -1)))))))
            else:
                p = self.tanh(self.linear3(self.tanh(self.linear2(self.relu(self.linear1((torch.cat((p.view(word_dim, -1), input[i+1]), 0)).view(1, -1)))))))

        for i in range(len(input2)-1):
            if i == 0:
                p2 = self.tanh(self.linear3(self.tanh(self.linear2(self.relu(self.linear1((torch.cat((input2[i], input2[i+1]), 0)).view(1, -1)))))))
            else:
                p2 = self.tanh(self.linear3(self.tanh(self.linear2(self.relu(self.linear1((torch.cat((p2.view(word_dim, -1), input2[i+1]), 0)).view(1, -1)))))))

        out = self.softmax(self.linear4(torch.cat((p, p2), -1)))
        return out
        