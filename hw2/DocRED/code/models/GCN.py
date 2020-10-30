import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from transformers import BertModel
from pygcn import GCN
from torch.nn import init
from torch.nn.utils import rnn
import logging

class BiLSTMGCN(nn.Module): 
	def __init__(self, config):
		super(BiLSTMGCN, self).__init__()
		self.config = config

		word_vec_size = config.data_word_vec.shape[0]
		self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
		self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

		self.word_emb.weight.requires_grad = False
		self.use_entity_type = True
		self.use_coreference = True
		self.use_distance = True

		hidden_size = 128
		input_size = config.data_word_vec.shape[1]
		if self.use_entity_type:
			input_size += config.entity_type_size
			self.ner_embed = nn.Embedding(7, config.entity_type_size, padding_idx=0)
		
		if self.use_coreference:
			input_size += config.coref_size
			self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

		# self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
		self.rnn = EncoderLSTM(input_size, hidden_size, 1, True, True, 1 - config.keep_prob, False)
		self.gcn = GCN(hidden_size*2, hidden_size*2, hidden_size*2, 0.1)
		self.linear_re = nn.Linear(hidden_size*2, hidden_size)

		if self.use_distance:
			self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
			self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
		else:
			self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)		 

	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping, 
							relation_mask, dis_h_2_t, dis_t_2_h, adj):
		sent = self.word_emb(context_idxs)
		if self.use_coreference:
			sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)
		
		if self.use_entity_type:
			sent = torch.cat([sent, self.ner_embed(context_ner)], dim=-1)
		context_output = self.rnn(sent, context_lens)
		context_output = self.gcn(context_output, adj)
		
		context_output = torch.relu(self.linear_re(context_output))

		start_re_output = torch.matmul(h_mapping, context_output)
		end_re_output = torch.matmul(t_mapping, context_output)

		if self.use_distance:
			s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
			t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
			predict_re = self.bili(s_rep, t_rep)
		else:
			predict_re = self.bili(start_re_output, end_re_output)

		return predict_re

class LockedDropout(nn.Module):
	def __init__(self, dropout):
		super().__init__()
		self.dropout = dropout

	def forward(self, x):
		dropout = self.dropout
		if not self.training:
			return x
		m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
		mask = Variable(m.div_(1 - dropout), requires_grad=False)
		mask = mask.expand_as(x)
		return mask * x
		
 
class EncoderLSTM(nn.Module):
	def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
		super().__init__()
		self.rnns = []
		for i in range(nlayers):
			if i == 0:
				input_size_ = input_size
				output_size_ = num_units
			else:
				input_size_ = num_units if not bidir else num_units * 2
				output_size_ = num_units
			self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
		self.rnns = nn.ModuleList(self.rnns)

		self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
		self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

		self.dropout = LockedDropout(dropout)
		self.concat = concat
		self.nlayers = nlayers
		self.return_last = return_last

		# self.reset_parameters()

	def reset_parameters(self):
		for rnn in self.rnns:
			for name, p in rnn.named_parameters():
				if 'weight' in name:
					p.data.normal_(std=0.1)
				else:
					p.data.zero_()

	def get_init(self, bsz, i):
		return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

	def forward(self, input, input_lengths=None):
		bsz, slen = input.size(0), input.size(1)
		output = input
		outputs = []
		if input_lengths is not None:
			lens = input_lengths.data.cpu().numpy()

		for i in range(self.nlayers):
			hidden, c = self.get_init(bsz, i)

			output = self.dropout(output)
			if input_lengths is not None:
				output = rnn.pack_padded_sequence(output, lens, batch_first=True)

			output, hidden = self.rnns[i](output, (hidden, c))


			if input_lengths is not None:
				output, _ = rnn.pad_packed_sequence(output, batch_first=True)
				if output.size(1) < slen: # used for parallel
					padding = Variable(output.data.new(1, 1, 1).zero_())
					output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
			if self.return_last:
				outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
			else:
				outputs.append(output)
		if self.concat:
			return torch.cat(outputs, dim=2)
		return outputs[-1]

		

		
