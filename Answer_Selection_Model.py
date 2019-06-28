import pickle
import random
import numpy as np
from scipy.stats import rankdata
import torch
#import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AnswerSelection(nn.Module):
    def __init__(self, args):
        super(AnswerSelection, self).__init__()
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.question_len = args.question_len
        self.answer_len = args.answer_len
        self.batch_size = args.batch_size

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.cnns = nn.ModuleList([nn.Conv1d(self.hidden_dim, 500, filter_size, stride=1, padding=filter_size-(i+1)) for i, filter_size in enumerate([1,3,5])])
        self.question_maxpool = nn.MaxPool1d(self.question_len, stride=1)
        self.answer_maxpool = nn.MaxPool1d(self.answer_len, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.init_weights()
        self.hiddenq = self.init_hidden(self.batch_size)
        self.hiddena = self.init_hidden(self.batch_size)

    def init_hidden(self, batch_len):
        return (Variable(torch.randn(2, batch_len, self.hidden_dim // 2)),
                Variable(torch.randn(2, batch_len, self.hidden_dim // 2)))

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, question, answer):
        question_embedding = self.word_embeddings(question)
        answer_embedding = self.word_embeddings(answer)
        q_lstm, self.hiddenq = self.lstm(question_embedding, self.hiddenq)
        a_lstm, self.hiddena = self.lstm(answer_embedding, self.hiddena)
        q_lstm = q_lstm.contiguous()
        a_lstm = a_lstm.contiguous()
        q_lstm = question_embedding
        a_lstm = answer_embedding
        q_lstm = q_lstm.view(-1,self.hidden_dim, self.question_len)
        a_lstm = a_lstm.view(-1,self.hidden_dim, self.answer_len)

        question_pool = []
        answer_pool = []
        for cnn in self.cnns:
            question_conv = cnn(q_lstm)
            answer_conv = cnn(a_lstm)
            question_max_pool = self.question_maxpool(question_conv)
            answer_max_pool = self.answer_maxpool(answer_conv)
            question_activation = F.tanh(torch.squeeze(question_max_pool))
            answer_activation = F.tanh(torch.squeeze(answer_max_pool))
            question_pool.append(question_activation)
            answer_pool.append(answer_activation)

        question_output = torch.cat(question_pool, dim=1)
        answer_output = torch.cat(answer_pool, dim=1)

        question_output = self.dropout(question_output)
        answer_output = self.dropout(answer_output)

        similarity = F.cosine_similarity(question_output, answer_output, dim=1)

        return similarity
    



