import torch
from GloVe import GloVe
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
	dictfile = './word_dict'
	embedfile = './GloVe'
	if(len(sys.argv) > 1):
		dictfile = sys.argv[1]
		embedfile = sys.argv[2]
	word_dict = pickle.load(open(dictfile, 'rb'))
	all_word = word_dict.keys()
	all_embed = pickle.load(open(embedfile, 'rb')).detach().numpy()
	while(1):
		word = input('Input Word: ')
		if(word not in all_word):
			print("This word is not trained!\n")
			continue
		idx = word_dict[word]
		# curr embed
		embed = all_embed[idx].reshape(1, -1)
		# cal simi
		d = cosine_similarity(embed, all_embed)[0]
		d = zip(all_word, d)
		d = sorted(d, key=lambda x:x[1], reverse=True)
		for w in d[:10]:
		    if len(w[0])<2:
		        continue
		    print(w)
		print('')