import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
from transformers import BertModel
import logging

class BERT(nn.Module): # TODO
	def __init__(self, config):
		super(BERT, self).__init__()
		self.config = config
		self.use_entity_type = True
		self.use_coreference = False # True
		self.use_distance = False # True
	
		hidden_size = 768
		self.hidden_size = hidden_size
		if self.use_entity_type:
			hidden_size += config.entity_type_size
			self.ner_embed = nn.Embedding(config.entity_type_size, config.entity_type_size, padding_idx=0) # Assume one label per entity
		
		if self.use_coreference:
			hidden_size += config.coref_size
			self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)

		self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
		self.linear_re = nn.Linear(hidden_size, hidden_size)

		if self.use_distance:
			self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
			self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
		else:
			self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)		 

	def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping, 
							relation_mask, dis_h_2_t, dis_t_2_h, wordpiece_map, wordpiece_mask):
		self.bert.eval()
		with torch.no_grad():
			batch_size = wordpiece_map.size(0)
			token_len = wordpiece_map.size(2)
			outputs = self.bert(context_idxs)[0]
			outputs = outputs.unsqueeze(2).expand(-1, -1, token_len, -1)
			wordpiece_map = wordpiece_map.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size)
			sent = (torch.gather(outputs, 1, wordpiece_map) * wordpiece_mask.unsqueeze(-1)).sum(2)
		if self.use_coreference:
			sent = torch.cat([sent, self.entity_embed(pos)], dim=-1)
		
		if self.use_entity_type:
			sent = torch.cat([sent, self.ner_embed(context_ner)], dim=-1)
		
		sent = torch.relu(self.linear_re(sent))
		start_re_output = torch.matmul(h_mapping, sent)
		end_re_output = torch.matmul(t_mapping, sent)

		if self.use_distance:
			s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
			t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
			predict_re = self.bili(s_rep, t_rep)
		else:
			predict_re = self.bili(start_re_output, end_re_output)

		return predict_re

		
 

		

