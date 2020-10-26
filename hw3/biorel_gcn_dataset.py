import json
import numpy as np
import os
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer

CLS = '[CLS]'
SEP = '[SEP]'
SPC = '#'
BLANK = 0
class BioRelGCNDataset(Dataset):
  def __init__(self, json_file,
                     config={}):
    ner2idx_file = config.get('ner2idx_file', 'ner2id.json')
    self.split = config.get('split', 'train')
    self.max_nchars_word = config.get('max_nchars_per_word', 16)
    self.max_nchars_sent = config.get('max_nchars_per_sent', 256)
    self.max_nwords = config.get('max_nwords', 100)
    self.data_dir = config.get('data_dir', '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw3/')
                
    if not os.path.isdir(self.data_dir):
      os.mkdir(self.data_dir)
 
    if not os.path.isdir(os.path.join(self.data_dir, 'prepro_data')):
      os.mkdir(os.path.join(self.data_dir, 'prepro_data'))
    self.out_path = os.path.join(self.data_dir, 'prepro_data')
    self.dep_parser = Predictor.from_path('https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz')
    self.load_data(os.path.join(self.data_dir, json_file))
    # if not os.path.isfile(os.path.join(self.data_dir, 'prepro_data', word_vec_file)):
    #   self.load_word_embed(self.embed_path, dimension=300, out_file=word_vec_file)            

    
  def load_data(self, data_file_name,
                char_limit = 16,
                max_length = 256):
    """
      Args:
        data_file_name: name of the data file in json format containing the annotations of the sentences           
          {
            'text': (sent str),
            'entities': [
               {'names': 
                (name_1): {
                  'is_mentioned': (bool),
                  'mentions': [[start_1, end_1], ..., [start_M, end_M]]
                  },
                (name 2): {...},
                ...,
                (name N): {...}]},
               {(another entity)}
             ]  
            }
      Returns:
        char_idxs: N x L x Kc Long Tensor
               [[[char2idx[c] for c in word_i^n] for i in range(L)] for n in range(N)]
        ner_idxs: N x L Long Tensor
              [[entity_i^n for i in range(L)] for n in range(N)]
        word_idxs: N x L Long Tensor
               [[word_i^n for i in range(L)] for n in range(N)]
        pos_idxs: N x L Long Tensor
             [[entity_order_i^n for i in range(L)] for n in range(N)]
        word_lens: N-dim Long Tensor storing the length of each word in characters
        sent_lens: N x L Long Tensor storing the length of each sent in words
    """
    ori_data = json.load(open(data_file_name))
    ner2id = json.load(open(os.path.join(self.out_path, 'ner2id.json')))
    dep_f = open(os.path.join(self.out_path, 'dep_parse.json'), 'w')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    
    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner_char = np.zeros((sen_tot, max_length*char_limit), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)
    adj_matrices = np.zeros((sen_tot, max_length, max_length)) 

    new_data = []
    for i, item in enumerate(ori_data):
      print('Instance {}'.format(i))
      # Use BERT's own tokenizer
      text = ' '.join([CLS, item['text'].replace(' ', SPC), SEP]) 
      tokenized_text = tokenizer.tokenize(text)
      sen_word_char = np.zeros(max_length*char_limit, dtype = np.int64)
      tokenized_text_no_spc = []
      j = 0
      start = 0
      for word in tokenized_text:
        if word == SPC:
          start += len(word)
          continue
        elif word == CLS or word == SEP:
          j += 1
          continue
        elif word[:2] == '##':
          word = word[2:] 
        elif word == 'cannot': # Dep parser breaks the word `cannot` into two by default
          sen_word_char[start:start+3] = j
          j += 1
          start += 3
          sen_word_char[start:start+3] = j
          tokenized_text_no_spc.append(word[:3])
          j += 1
          start += 3
          tokenized_text_no_spc.append(word[3:])
        elif j < max_length:
          sen_word_char[start:start+len(word)] = j
          j += 1
          start += len(word)
          tokenized_text_no_spc.append(word)
      
      # print(tokenized_text_no_spc)
      # Convert tokens to ids
      indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_no_spc)[:max_length]
      sen_word[i, :len(indexed_tokens)] = deepcopy(indexed_tokens)
      
      for j in range(j + 1, max_length):
        sen_word[i][j] = BLANK
        start += len(word)                      

      adj_matrix, deps, heads = self.parse_sent(tokenized_text_no_spc, max_length)
      parse_dict = {'dependencies': deps, 'heads': heads}
      dep_f.write('{}\n'.format(json.dumps(parse_dict)))
      adj_matrices[i, :adj_matrix.shape[0], :adj_matrix.shape[1]] = deepcopy(adj_matrix) 

      entities = item['entities']
      mentions = []
      for idx in range(len(entities)):
        for k in entities[idx]['names']:
          for i_m, em in enumerate(entities[idx]['names'][k]['mentions']):
            start = sen_word_char[em[0]]
            end = sen_word_char[em[1]-1]+1
            sen_ner_char[start:end] = ner2id[entities[idx]['label']]
            entities[idx]['names'][k]['mentions'][i_m][0] = int(start)
            entities[idx]['names'][k]['mentions'][i_m][1] = int(end)
            mentions.append([start, end])

      mentions = sorted(mentions, key=lambda x:x[0])
      for i_m, mention in enumerate(mentions, 1):
        sen_pos[i][mention[0]:mention[1]] = i_m
      new_data.append({'text': item['text'], 'entities': entities})
        
    np.save(os.path.join(self.out_path, self.split+'_gcn_word.npy'), sen_word)
    np.save(os.path.join(self.out_path, self.split+'_gcn_pos.npy'), sen_pos)
    np.save(os.path.join(self.out_path, self.split+'_gcn_ner.npy'), sen_ner)
    np.save(os.path.join(self.out_path, self.split+'_gcn_char.npy'), sen_char)
    np.save(os.path.join(self.out_path, self.split+'_gcn_depmat.npy'), adj_matrices)
    json.dump(new_data, open(os.path.join(self.out_path, self.split+'_gcn_new.json'), 'w'), indent=4, sort_keys=True)

  def parse_sent(self,  
                 sent,
                 max_length = 256):
                # Parse the sentence
                merged_tokens = []
                token_to_merged_token = np.zeros(max_length, dtype = np.int64)  
                j = 0
                for idx, token in enumerate(sent[:max_length]):
                        if token[:2] == '##':
                          token_to_merged_token[idx] = j
                          merged_tokens[-1] += token[2:]
                        elif token == CLS or token == SEP:
                          continue
                        else:
                          token_to_merged_token[idx] = j
                          j += 1
                          merged_tokens.append(token)
                        
                merged_token_to_token = [np.nonzero(token_to_merged_token==merge_idx)[0] for merge_idx in range(len(merged_tokens))]

                parsed_sent = self.dep_parser.predict(' '.join(merged_tokens))
                predicted_dep = parsed_sent['predicted_dependencies']
                predicted_heads = parsed_sent['predicted_heads']
                # print('predicted_dep: {}'.format(predicted_dep))
                # print(merged_tokens)
                # print('predicted heads: {}, len: {}, len(merged_tokens): {}'.format(predicted_heads, len(predicted_heads), len(merged_tokens)))
                A = np.zeros((max_length, max_length))
                for t_idx, token in enumerate(sent[:max_length]):
                        if token == CLS or token == SEP:
                                continue
                        merge_t_idx = token_to_merged_token[t_idx]
                        merge_h_idx = predicted_heads[merge_t_idx]
                        if merge_h_idx >= len(merged_token_to_token):
                          print('Warning: idx {} exceeds sentence len {}'.format(merge_h_idx, len(merged_token_to_token)))
                          merge_h_idx = len(merged_token_to_token) - 1
                        h_idxs = deepcopy(merged_token_to_token[merge_h_idx])
                        for h_idx in h_idxs:
                          if h_idx != t_idx:
                            A[h_idx, t_idx] = 1. 
                return A, predicted_dep, predicted_heads
        

  # def get_batch(self):
  def __getitem__(self, idx):
    return torch.LongTensor(self.word_idxs),\
           torch.LongTensor(self.char_idxs),\
           torch.LongTensor(self.ner_idxs),\
           torch.LongTensor(self.pos_idxs),\
           torch.LongTensor(self.word_lens),\
           torch.LongTensor(self.char_lens)
  
  def clean(self, word):
    for c in PUNCT:
      word = word.lower().lstrip(c).rstrip(c)
    return word

if __name__ == '__main__':
  train_set = BioRelGCNDataset('1.0alpha7.train.json', {'split': 'train'})
  dev_set = BioRelGCNDataset('1.0alpha7.dev.json', {'split': 'dev'})
