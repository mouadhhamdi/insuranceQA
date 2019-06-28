import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, args, word_vectors=None): # In QA-LSTM model, embedding weights is fine-tuned
        super(WordEmbedding, self).__init__()
        if args.use_glove:
            self.embedding = nn.Embedding.from_pretrained(word_vectors)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embd_size)

    def forward(self, x):
        return self.embedding(x)


class QA_LSTM(nn.Module):
    def __init__(self,args, word_vectors=None):
        super(QA_LSTM, self).__init__()
        self.word_embd = WordEmbedding(args, word_vectors)
        self.shared_lstm = nn.LSTM(args.embd_size, args.hidden_size, batch_first=True, bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        # embedding
        q = self.word_embd(q) # (bs, L, E)
        a = self.word_embd(a) # (bs, L, E)

        # LSTM
        q, _h = self.shared_lstm(q) # (bs, L, 2H)
        a, _h = self.shared_lstm(a) # (bs, L, 2H)

        # mean
        # q = torch.mean(q, 1) # (bs, 2H)
        # a = torch.mean(a, 1) # (bs, 2H)
        # maxpooling
        q = torch.max(q, 1)[0] # (bs, 2H)
        a = torch.max(a, 1)[0] # (bs, 2H)

        return self.cos(q, a) # (bs,)

