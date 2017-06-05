import argparse

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from unicodedata import bidirectional

parser = argparse.AugmentParser()

parser.add_argument('--data', dest='data', help='path to data folder', default="./data")
parser.add_argument('--hidden_size', dest='hsize', type=int, help='hidden size')
parser.add_argument('--model', dest='model', help='model type rnn or rntn', default='rnn')
parser.add_argument('--seed', dest='seed', type=int, help='seed', default=1234567890)
parser.add_argument('--cuda', action='store_true', help='use GPU for faster training')

args = parser.parse_args()

cuda = args.cuda

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)        

class BiLSTM_CRF(nn.Module):
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim)))
    
    def _forward_alg(self, feats):
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.) 
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
    
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
                
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
            
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        best_path = [best_tag_id]
        
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        self.hidden = self.init_hidden()
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    
    def forward(self, sentence):
        self.hidden = self.init_hidden()
        lstm_feats = self._get_lstm_features(sentence)
        
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

class SemCom(nn.Module):
    
    def __init__(self):
        super(SemCom, self).__init__()

class RecNN(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super(RecNN, self).__init__()
        
class SimMea(nn.Module):
    
    def __init__(self):
        super(SimMea, self).__init__()
        
def train(model):
    n = correct = 0.0
    
def test():