import numpy as np
import collections
import zipfile
import random


class Dataset:
    """Do some managements of training data
    
    Attributes:
        words:         all the words in training file
        count:         a list of [[word: count], ... ...], the length is word_num
        word_dict:     the dict of words with the order of count, {word: idx}
        rev_word_dict: reverse word_dict, {idx: word}
        data:          a list of idx corresponding to words 
        sample_data:   sampleing table for nagative sampling
        skip_index:    index of the first data of one skip in one generation
    """
    
    def __init__(self, filename, word_num):
        """Init Dataset class with input file and expected word_num
        
        Args:
            filename: filename of input training data
            word_num: number of expected number of training words
        """
        self.filename = filename
        self.word_num = word_num
        self.words = '' 
        self.count = [['UNK', -1]]
        self.word_dict = dict()
        self.rev_word_dict = dict()
        self.data = list()
        self.skip_index = 0 
        self.sample_table = list() 
        self.build_dataset()
        self.build_sample_table()
    
    def build_dataset(self):
        """Build the dataset and get words, count, word_dict and rev_word_dict"""
        print('building dataset ... ...')
        # read words from zip file
        print('\t reading words ... ...')
        with zipfile.ZipFile(self.filename) as f:
            self.words = np.array(f.read(f.namelist()[0]).decode(encoding='utf-8').split())
        # count the appearence of each word 
        print('\t counting words ... ...')
        self.count.extend(collections.Counter(self.words).most_common(self.word_num-1))
        # construct word_dict
        print('\t constructing word dict ... ...')
        for w, _ in self.count:
            self.word_dict[w] = len(self.word_dict)
        # transfer word into number, store in list 'data'
        print('\t word to number ... ...')
        unk_count = 0
        for w in self.words:
            index = self.word_dict.get(w, 0)
            if index == 0:
                unk_count += 1
            self.data.append(index)
        self.count[0][1] = unk_count
        # reverse word_dict
        print('\t reverse word _dict')
        self.rev_word_dict = dict(zip(self.word_dict.values(), self.word_dict.keys()))
        
    def gen_batch(self, batch_size, skip_window, num_skips):
        """Generate one batch for batch training
        
        Args:
            batch_size:  size of one batch
            skip_window: length of one skip window on one side of target [skip_windos, target ,skip_window]
            num_skips:   number of selected context word in one skip
            
        Returns:
            x, y: np.array of x and y in this batch
        """
        # assert the relation between paramters
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        # init the shape of x, y
        x = np.ndarray(shape=(batch_size), dtype=np.int32)
        y = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1 
        context_words = [w for w in range(span) if w != skip_window]
        buffer = collections.deque(maxlen=span) # data of current [skip_windos, target ,skip_window]
        if self.skip_index + span > len(self.data):
            self.skip_index = 0
        buffer.extend(self.data[self.skip_index: self.skip_index + span])
        for i in range(batch_size // num_skips):
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                x[i * num_skips + j] = buffer[skip_window]
                y[i * num_skips + j, 0] = buffer[context_word]
            if self.skip_index == len(self.data)-span:
                buffer.extend(self.data[0:span])
                self.skip_index = 0
            else:
                buffer.append(self.data[self.skip_index+span])
                self.skip_index += 1
        return x, y
    
    def build_sample_table(self):
        """Build the sampling table for negative sampling"""
        print('build sample table ... ...')
        table_size = 1e8
        count_list = list(dict(self.count).values())
        numerator = np.array(count_list)**0.75
        denominator = sum(count_list)**0.75
        ratio = numerator / denominator
        width = np.round(ratio * table_size)
        for index, w in enumerate(width):
            self.sample_table += [index] * int(w)
        self.sample_table = np.array(self.sample_table)
            
    def get_neg_sample(self, x_len, neg_cnt):
        """Get negatiee samples of current batch
        
        Args: 
            x_len:   length of x in this batch, actually the batch size
            neg_cnt: number of negative samples for one x
            
        Returns:
            neg: negative samples selected by random for this batch
        """
        neg = np.random.choice(self.sample_table, size=(x_len, neg_cnt)).tolist()
        return neg