import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return F.relu(self.gc2(x, adj))

class HGCN(nn.Module):
		""" Hetereogeneous graph convolutional net """
    def __init__(self, nfeat, nhid, nclass, nrel, dropout):
        super(GCN, self).__init__()

        self.gc1 = [GraphConvolution(nfeat, nhid) for r in range(nrel)]
        self.gc2 = [GraphConvolution(nhid, nclass) for r in range(nrel)]
        self.dropout = dropout
				self.nrel = nrel

    def forward(self, x, adj):
				xs = [] 	
				adjs = [(adj == r).type(torch.FloatTensor) for r in range(1, self.nrel+1)]
				for r in range(1, self.nrel+1):
					x = F.relu(self.gc1[r](x, adjs[r]))
	        x = F.dropout(x, self.dropout, training=self.training)
					xs.append(x)  
				x = np.stack(xs, dim=0).sum(dim=0)      
				
				xs = []
				for r in range(1, self.nrel+1):
					x = F.relu(self.gc2[r](x, adjs[r]))
					x = F.dropout(x, self.dropout, training=self.training)
					xs.append(x)
				x = np.stack(xs, dim=0).sum(dim=0)
        # return F.log_softmax(x, dim=1)
        return x

        
