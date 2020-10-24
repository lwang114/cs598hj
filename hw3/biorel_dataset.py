import json
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize

UNK = '<UNK>'
NULL = '<NULL>'
PUNCT = [' ', ',', '.', '(', ')']
class BioRelDataset(Dataset):
  def __init__(self, json_file,
               config={}):
    word2idx_file = config.get('word2idx_file', 'word2idx.json')
    ner2idx_file = config.get('ner2idx_file', 'ner2idx.json') 
    char2idx_file = config.get('char2idx_file', 'char2idx.json')
    self.split = config.get('split', 'train')
    self.max_nchars_word = config.get('max_nchars_per_word', 16)
    self.max_nchars_sent = config.get('max_nchars_per_sent', 512)
    self.max_nwords = config.get('max_nwords', 100)
    self.data_dir = config.get('data_dir', '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw3/')
    
    self.word2idx = {UNK:0}
    self.ner2idx = {NULL:0}
    self.char2idx = {NULL:0} 
    self.load_data(os.path.join(self.data_dir, json_file),
              os.path.join(self.data_dir, word2idx_file),
              os.path.join(self.data_dir, ner2idx_file),
              os.path.join(self.data_dir, char2idx_file))

  def load_data(self, json_file,
                word2idx_file,
                ner2idx_file,
                char2idx_file):
    """
      Args:
        json_file: name of the json file containing the annotations of the sentences       
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
    with open(json_file, 'r') as f:
      data_dicts = json.load(f)

    if not os.path.isfile(word2idx_file) or not os.path.isfile(ner2idx_file) or not os.path.isfile(char2idx_file):
      for data_dict in data_dicts:
        text = data_dict['text'].split()
        entities = data_dict['entities']
        
        for word in text:
          word = self.clean(word)
          if not word in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
          
          for char in word:
            if not char in self.char2idx:
              self.char2idx[char] = len(self.char2idx)
          
        for entity in entities: # Create ner2idx
          entity_label = entity['label']
          if not entity_label in self.ner2idx:
            self.ner2idx[entity_label] = len(self.ner2idx)

      with open(word2idx_file, 'w') as word2idx_f,\
           open(char2idx_file, 'w') as char2idx_f,\
           open(ner2idx_file, 'w') as ner2idx_f:
          json.dump(self.word2idx, word2idx_f, indent=4, sort_keys=True) 
          json.dump(self.char2idx, char2idx_f, indent=4, sort_keys=True)
          json.dump(self.ner2idx, ner2idx_f, indent=4, sort_keys=True) 
    else:
      self.word2idx = json.load(open(word2idx_file))
      self.char2idx = json.load(open(char2idx_file))
      self.ner2idx = json.load(open(ner2idx_file))

    print('Size of vocabs={}'.format(len(self.word2idx)))
    print('Number of entity types={}'.format(len(self.char2idx)))
    print('Number of character types={}'.format(len(self.ner2idx)))

    word_idxs = []
    char_idxs = []
    ner_idxs = []
    pos_idxs = []
    word_lens = []
    char_lens = []
    
    cur_word_idxs = np.zeros(self.max_nwords, dtype=np.int)
    cur_char_idxs = np.zeros((self.max_nwords, self.max_nchars_word), dtype=np.int)
    cur_char_lens = np.zeros(self.max_nwords, dtype=np.int)
    for ex, data_dict in enumerate(data_dicts):
      print('{} instance {}'.format(self.split, ex))
      text = data_dict['text'].split()
      entities = data_dict['entities']

      # Load word and character labels and the word spans
      spans = np.zeros(len(text), dtype=np.int)
      start = 0
      for i_w, w in enumerate(text):
        if i_w >= self.max_nwords or start >= self.max_nchars_sent:
          break
          
        spans[i_w] = min(start, self.max_nchars_sent)
        start = spans[i_w] + len(w) + 1
        cur_char_lens[i_w] = min(len(w), self.max_nchars_word)
      
        w = self.clean(w[:cur_char_lens[i_w]])  
        cur_word_idxs[i_w] = self.word2idx.get(w, 0)
        for i_c, c in enumerate(w):
          cur_char_idxs[i_w, i_c] = self.char2idx.get(c, 0)

      # Load character-level entity labels
      labels_char_level = np.zeros(self.max_nchars_sent)
      for i_e, entity in enumerate(entities):
        entity_label = entity['label']
        for name, name_info in entity['names'].items():
          if name_info['is_mentioned']:
            mentions = name_info['mentions']    
            for mention in mentions:
              start, end = mention[0], mention[1]
              end = start + min(end - start, self.max_nchars_sent)
              labels_char_level[start:end] = self.ner2idx.get(entity_label, 0)

      # Convert the character-level entity labels to word-level entity labels
      cur_ner_idxs = np.zeros(self.max_nwords, dtype=np.int)
      cur_pos_idxs = np.zeros(self.max_nwords, dtype=np.int) 
      i_e = 1
      for i_w, word in enumerate(text):
          # Sentence lengths in words
          if i_w >= self.max_nwords:
            break 

          # Entity ids
          entity_id = labels_char_level[spans[i_w]]
          cur_ner_idxs[i_w] = entity_id
          
          # Position ids
          if not entity_id == self.ner2idx[NULL]:
            cur_pos_idxs[i_w] = i_e
            i_e += 1

      word_idxs.append(cur_word_idxs)
      word_lens.append(min(len(text), self.max_nwords))
      char_idxs.append(cur_char_idxs)
      char_lens.append(cur_char_lens)
      ner_idxs.append(cur_ner_idxs)
      pos_idxs.append(cur_pos_idxs)

    self.word_idxs = np.asarray(word_idxs)
    self.char_idxs = np.asarray(char_idxs)
    self.ner_idxs = np.asarray(ner_idxs)
    self.pos_idxs = np.asarray(pos_idxs)
    self.word_lens = np.asarray(word_lens)
    self.char_lens = np.asarray(char_lens)
    if not os.path.isdir(os.path.join(self.data_dir, 'prepro_data')):
      os.mkdir(os.path.join(self.data_dir, 'prepro_data'))
    np.save('{}/{}_word.npy'.format(self.data_dir, self.split), self.word_idxs)
    np.save('{}/{}_char.npy'.format(self.data_dir, self.split), self.char_idxs)
    np.save('{}/{}_pos.npy'.format(self.data_dir, self.split), self.pos_idxs)
    np.save('{}/{}_ner.npy'.format(self.data_dir, self.split), self.ner_idxs)
    
  # def get_batch(self): # TODO
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
  
  train_set = BioRelDataset('1.0alpha7.train.json', {'split': 'train'})
  dev_set = BioRelDataset('1.0alpha7.dev.json', {'split': 'dev'})
