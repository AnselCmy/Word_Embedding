## GloVe by PyTorch

### Chen Mingyang / Oct. 16 2018

 #### Implementation

1. Dataset

   ```python
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
           
       def read_file(self):
           """Read file by filename into words"""
               
       def build_dataset(self):
           """Build the dataset and get count, word_dict and rev_word_dict"""
       
       def build_comat(self):
           """Build co-occurrence matrix"""
          
       def gen_batch(self, batch):
           """Generate batch for batch learning"""
   ```

2. GloVe

   ```python
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
       
       def f(self, xx):
           """Implementation of weright funtion"""
   
       def forward(self, x, y):
           """Process of fowarding from imput word pair to cost function J in batch form"""
           
       def train(self):
           """Start training the model and embedding"""
   ```

#### Train and Test

1. Train GloVe 

   ```bash
   python ./train.py
   ```

   You can set hyper paramters in the main of train.py. The default output embedding is './GloVe', and default dictionary is './word_dict'

2. Test trained model

   The embedding model file and word dictinary file can be transformed as argvs, 

   ```bash
   python ./test.py word_dict GloVe
   ```

   and "glove" and "word_dict" are default argvs. 

   ```bash
   python ./test.py
   ```

3. Pre-trained embedding

   File 'GloVe_1k_11h' and 'word_dict_1k_11h' is the pre-trained embedding and word dictionary of 1k words from text8. Try to test it.

   ```Bash
   python ./test.py word_dict_1k_11h GloVe_1k_11h
   ```

#### References

+ GloVe: Global Vectors for Word Representation https://nlp.stanford.edu/projects/glove/
+ GloVe:另一种Word Embedding方法 http://www.pengfoo.com/post/machine-learning/2017-04-11

+ Glove http://vsooda.github.io/2016/04/06/Glove/

  



