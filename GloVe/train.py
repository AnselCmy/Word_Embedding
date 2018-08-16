from GloVe import GloVe

if __name__ == '__main__':
	filename='./text8.zip'
	word_num=1000
	context_size=3 
	batch=8
	x_max=3
	alpha=0.75
	embed_dim=20
	epoch=1
	lr=0.001
	outfile='./GloVe'
	dictfile='./word_dict'
	glove = GloVe(filename, word_num, context_size, batch,
                  x_max, alpha, embed_dim, epoch, lr,
                  outfile, dictfile)
	glove.train()