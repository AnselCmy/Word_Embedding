import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    """Class of skip gram model 
    
    Attributes:
        embed_num:   size of the dictionary of embeddings
        embed_dim:   the size of each embedding vector
        embed:       weight of networks for input to hidden
        embed_prime: weight of networks for hidden to output
    """
    
    def __init__(self, embed_num, embed_dim):
        """Init skip gram model
        
        Args:
            embed_num:   size of the dictionary of embeddings
            embed_dim:   the size of each embedding vector
        """
        super(SkipGram, self).__init__()
        self.embed_num = embed_num 
        self.embed_dim = embed_dim  
        # init embeddings 
        self.embed = nn.Embedding(embed_num, embed_dim, sparse=True)
        self.embed_prime = nn.Embedding(embed_num, embed_dim, sparse=True)
        self.embed.weight.data.uniform_(-0.5 / self.embed_dim, 0.5 / self.embed_dim)
        self.embed_prime.weight.data.uniform_(-0, 0)
        
    def forward(self, x, y, neg):
        """Forward of this netword
        
        Args: 
            x:   training x
            y:   training y
            neg: negative sample 
        
        Returns:
            Optimization target function
        """
        embed_x = self.embed(x)
        embed_y = self.embed_prime(y)
        embed_neg = self.embed_prime(neg)
        score = torch.mul(embed_x, embed_y).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        neg_score = torch.bmm(embed_neg, embed_x.unsqueeze(2)).squeeze() # batch matrix matrix
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score))