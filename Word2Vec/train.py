from Word2Vec import Word2Vec

if __name__ == '__main__':
	filename='./text8.zip'
	word_num=1000
	batch_size=8
	skip_window=2
	num_skips=2
	embed_dim=100
	epoch=1
	lr=0.025
	neg_cnt=5
	outfile = './skip_gram'
	dicfile = './word_dict'
	w2v = Word2Vec(filename, word_num, 
				   batch_size, skip_window, 
				   num_skips, embed_dim, 
				   epoch, lr, 
				   neg_cnt,
				   outfile, dicfile)
	w2v.train()