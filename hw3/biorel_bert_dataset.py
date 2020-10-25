import json
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer

CLS = '[CLS]'
SEP = '[SEP]'
SPC = '#'
BLANK = 0
PUNCT = [' ', ',', '.', '(', ')']
class BioRelBertDataset(Dataset):
  def __init__(self, json_file,
                     config={}):
    ner2idx_file = config.get('ner2idx_file', 'ner2id.json') 
    char2idx_file = config.get('char2idx_file', 'char2id.json')
    rel2idx_file = config.get('rel2idx_file', 'rel2id.json')
    word_vec_file = config.get('word_vec_file', 'vec.npy') # XXX Placeholder
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    

    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner_char = np.zeros((sen_tot, max_length*char_limit), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

    new_data = []
    for i, item in enumerate(ori_data):
      # Use BERT's own tokenizer
      text = ' '.join([CLS, item['text'].replace(' ', SPC), SEP]) # TODO Check the effect of adding SPC 
      tokenized_text = tokenizer.tokenize(tokenized_text)
      print(tokenized_text)
      sen_word_char = np.zeros(max_length*char_limit, dtype = np.int64)
      j = 0
      start = 0
      for word in tokenized_text:
        if word == SPC:
          start += len(word)
          continue
        elif word == CLS or word == SEP:
          j += 1
          continue
        elif word[0:2] == '##':
          word = word[2:] 
                                
        if j < max_length:
          sent_word_char[start:start+len(word)] = j
          j += 1
          sen_word[i] = tokenizer.convert_tokens_to_ids([word for word in tokenized_text if word != SPC])[:max_length]
      
      for j in range(j + 1, max_length):
        sen_word[i][j] = BLANK
        start += len(word)      
     
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
        
    np.save(os.path.join(self.out_path, self.split+'_word.npy'), sen_word)
    np.save(os.path.join(self.out_path, self.split+'_pos.npy'), sen_pos)
    np.save(os.path.join(self.out_path, self.split+'_ner.npy'), sen_ner)
    np.save(os.path.join(self.out_path, self.split+'_char.npy'), sen_char)
    json.dump(new_data, open(os.path.join(self.out_path, self.split+'_new.json'), 'w'), indent=4, sort_keys=True)

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
  train_set = BioRelDataset('1.0alpha7.train.json', {'split': 'train', 'data_dir': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw3/bert/'})
  dev_set = BioRelDataset('1.0alpha7.dev.json', {'split': 'dev', 'data_dir': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw3/bert/'})
