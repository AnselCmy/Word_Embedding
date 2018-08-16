import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
from Dataset import Dataset
import pickle


class GloVe:
    """Main part of GloVe model for training 

    Attributes:
        filename:     filename of input training data
        word_num:     number of expected number of training words
        context_size: considered size of context of one word  
        batch:        number of training samples in one batch
        x_max:        threshold for weight function
        alpha:        parameter for weight function 
        embed_dim:    size of each embedding vector
        epoch:        number of iteration 
        lr:           learning rate
        outfile:      filename of output trained glove embedding 
        dictfile:     filename of word dict for storing word dict
    """

    def __init__(self, filename='./text8.zip', word_num=100, context_size=3, batch=8,
                 x_max=3, alpha=0.75, embed_dim=10, epoch=10, lr=0.001,
                 outfile='./GloVe', dictfile='./word_dict'):
        """Init this GloVe model"""
        self.batch = batch
        self.x_max = x_max
        self.alpha = alpha
        self.epoch = epoch
        self.lr = lr
        self.outfile = outfile
        if(os.path.exists(outfile)):
            self.embed = pickle.load(open(self.outfile, 'rb'))
        else:
            self.embed = Variable(torch.from_numpy(np.random.normal(0, 0.01, (word_num, embed_dim))), requires_grad = True)
        self.bias = Variable(torch.from_numpy(np.random.normal(0, 0.01, word_num)), requires_grad = True)
        self.dataset = Dataset(filename, word_num, context_size)
        if(not os.path.exists(dictfile)):
            pickle.dump(self.dataset.word_dict, open(dictfile, 'wb'))
        # self.dataset = dataset
        self.optimizer = optim.Adam([self.embed, self.bias], lr = lr)
    
    
    def f(self, xx):
        """Implementation of weright funtion"""
        return torch.DoubleTensor([(x / self.x_max)**self.alpha if x < self.x_max else 1 for x in xx ])
    
    def forward(self, x, y):
        """Process of fowarding from imput word pair to cost function J in batch form"""
        embed_x = self.embed[x]
        embed_y = self.embed[y]
        bias_x = self.bias[x]
        bias_y = self.bias[y]
        coval = self.dataset.comat[x, y]
        j = torch.bmm(embed_x.unsqueeze(1), embed_y.unsqueeze(1).transpose(1,2))
        j = j.squeeze() + bias_x + bias_y
        j = (j - torch.from_numpy(np.log(coval)))**2
        j = sum(torch.mul(j, self.f(coval)))
        return j
        
    def train(self):
        """Start training the model and embedding"""
        batch_num = len(self.dataset.data)//self.batch
        bar = tqdm(range(self.epoch * batch_num))
        # bar = range(self.epoch * batch_num)
        for i in bar:
            x, y = self.dataset.gen_batch(self.batch)
            x = Variable(torch.LongTensor(x))
            y = Variable(torch.LongTensor(y))
            self.optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            self.optimizer.step()
            if(i % 10000 == 0):
                pickle.dump(self.embed, open(self.outfile, 'wb'))
            bar.set_description("Loss: %0.8f" % loss.data)
        pickle.dump(self.embed, open(self.outfile, 'wb'))