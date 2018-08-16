import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import collections
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Dataset import Dataset
from SkipGram import SkipGram


class Word2Vec():
    """Main class for word2vec model
    
    Attributes:
        filename:    filename of input training data
        word_num:    number of expected number of training words
        batch_size:  number of training samples in one batch
        skip_window: length of one skip window on one side of target [skip_windos, target ,skip_window]
        num_skips:   number of selected context word in one skip
        embed_dim:   size of each embedding vector
        epoch:       number of iteration 
        lr:          learning rate
        neg_cnt:     number of negative samples for one x
        outfile:     filename of output trained skip gram model 
        dictfile:    filename of word dict for storing word dict
    """

    def __init__(self, filename='./text8.zip', word_num=200, 
                 batch_size=8, skip_window=2, num_skips=2,
                 embed_dim=10,
                 epoch=100, lr=0.025, 
                 neg_cnt=5,
                 outfile='./skip_gram', dictfile='./word_dict'):
        """Init this word2vec model"""
        # params about dataset
        self.batch_size = batch_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        # params about skip gram
        self.embed_num = word_num
        self.embed_dim = embed_dim
        # params about learning
        self.epoch = epoch
        self.lr = lr
        self.neg_cnt = neg_cnt
        # dataset
        self.dataset = Dataset(filename, word_num)
        if(not os.path.exists(dictfile)):
            pickle.dump(self.dataset.word_dict, open(dictfile, 'wb'))
        # skip gram
        self.outfile = outfile
        if(os.path.exists(outfile)):
            self.skip_gram = pickle.load(open(self.outfile, 'rb'))
        else:
            self.skip_gram = SkipGram(word_num, embed_dim)
        # optimizer
        self.optimizer = optim.SGD(self.skip_gram.parameters(), lr=self.lr)
        
    def train(self):
        """Start training the model and embedding"""
        batch_num = len(self.dataset.data)-2*self.skip_window
        bar = tqdm(range(self.epoch * batch_num))
        for i in bar:
            x, y = self.dataset.gen_batch(self.batch_size, self.skip_window, self.num_skips)
            neg = self.dataset.get_neg_sample(len(x), self.neg_cnt)
            x = Variable(torch.LongTensor(x))
            y = Variable(torch.LongTensor(y))
            neg = Variable(torch.LongTensor(neg))
            # backprop
            self.optimizer.zero_grad()
            loss = self.skip_gram.forward(x, y, neg)
            loss.backward()
            self.optimizer.step()
            # set output
            if(i % 200000 == 0):
                pickle.dump(self.skip_gram, open(self.outfile, 'wb'))
            bar.set_description("Loss: %0.8f" % loss.data)
            # pickle
        pickle.dump(self.skip_gram, open(self.outfile, 'wb'))