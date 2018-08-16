import numpy as np
import collections
import zipfile


class Dataset:
    """Do some managements of training data
    
    Attributes:
        filename:      filename of input training data
        word_num:      number of expected number of training words
        context_size:  considered size of context of one word  
        words:         all the words in training file
        count:         a list of [[word: count], ... ...], the length is word_num
        word_dict:     the dict of words with the order of count, {word: idx}
        rev_word_dict: reverse word_dict, {idx: word}
        data:          a list of idx corresponding to words 
        commat:        co-occurrence matrix 
        comat_nz:      none-zero position in commat
    """

    def __init__(self, filename='./text8.zip', word_num=100, context_size=3):
        """Init Dataset class with input file and expected word_num, context_size
        
        Args:
            filename:     filename of input training data
            word_num:     number of expected number of training words
            context_size: considered size of context of one word  
        """
        self.filename = filename
        self.word_num = word_num
        self.context_size = context_size
        self.words = list()
        self.count = [['UNK', -1]]
        self.word_dict = dict()
        self.rev_word_dict = dict()
        self.data = list()
        self.comat = np.zeros((word_num, word_num))
        self.comat_nz = np.array([])
        self.read_file()
        self.build_dataset()
        self.build_comat()
        
    def read_file(self):
        """Read file by filename into words"""
        print('reading words ... ...')
        with zipfile.ZipFile(self.filename) as f:
            self.words = np.array(f.read(f.namelist()[0]).decode(encoding='utf-8').split())
            
    def build_dataset(self):
        """Build the dataset and get count, word_dict and rev_word_dict"""
        print('counting words ... ...')
        self.count.extend(collections.Counter(self.words).most_common(self.word_num-1))
        # construct word_dict
        print('constructing word dict ... ...')
        for w, _ in self.count:
            self.word_dict[w] = len(self.word_dict)
        # transfer word into number, store in list 'data'
        print('word to index ... ...')
        unk_count = 0
        for w in self.words:
            index = self.word_dict.get(w, 0)
            if index == 0:
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        # reverse word_dict
        print('reverse word dict')
        self.rev_word_dict = dict(zip(self.word_dict.values(), self.word_dict.keys()))
    
    
    def build_comat(self):
        """Build co-occurrence matrix"""
        print('building co-occurrences matrix')
        for i in range(len(self.data)):
            for j in range(1, self.context_size+1):
                if i-j > 0:
                    self.comat[self.data[i], self.data[i-j]] += 1.0/j
                elif i+j < len(self.data):
                    self.comat[self.data[i], self.data[i+j]] += 1.0/j
        self.comat_nz = np.transpose(np.nonzero(self.comat))
                    
    def gen_batch(self, batch):
        """Generate batch for batch learning"""
        batch_idx = np.random.choice(np.arange(len(self.comat_nz)), size=batch, replace=False)
        x = []
        y = []
        for i in batch_idx:
            pos = tuple(self.comat_nz[i])
            x.append(pos[0])
            y.append(pos[1])
        return x, y 