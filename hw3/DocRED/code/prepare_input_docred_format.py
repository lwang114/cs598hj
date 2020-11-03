import spacy
import json
import argparse
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

nlp = spacy.load('en_core_sci_sm')

UNK = '<UNK>'
def int_overlap(a1, b1, a2, b2):
    """Checks whether two intervals overlap"""
    if b1 < a2 or b2 < a1:
        return False
    return True

class Token:
    def __init__(self, text, start_char, end_char):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char


class Sentence:
    def __init__(self, data):
        self.data = data
        self.tokens = []
        self.whitespaces = []
        self.sentence = []
        self.tokenize()
        self.ner_list = []
        self.ent_ids_to_ner_ids = {}
        self.get_entities()
        self.clusters = []
        self.get_clusters()
        self.relations = []
        self.get_relations()

    def tokenize(self):
        sent = self.data['text']
        doc = nlp(sent, disable=['tagger', 'parser', 'ner', 'textcat'])
        char_pos = 0
        for token in doc:
            end_pos = char_pos + len(token.text) - 1
            self.sentence.append(token.text)
            self.tokens.append(Token(token.text, char_pos, end_pos))
            self.whitespaces.append(token.whitespace_)
            char_pos = end_pos + 1
            if token.whitespace_:
                char_pos += 1

    def get_entities(self):
        for ent_id, ent in enumerate(self.data['entities']):
            if not ent['is_mentioned']:
                continue

            ent_type = ent['label']

            for entity in ent['names'].values():
                if not entity['is_mentioned']:
                    continue

                for mention in entity['mentions']:
                    start = mention[0]
                    end = mention[1] - 1
                    ne_start = None
                    ne_end = None
                    for idx, token in enumerate(self.tokens):
                        if int_overlap(start, end,
                                       token.start_char, token.end_char):
                            if ne_start is None:
                                ne_start = idx
                            ne_end = idx

                    ne = (ne_start, ne_end, ent_type)
                    if ne not in self.ner_list:
                        self.ner_list.append(ne)
                        ner_id = len(self.ner_list) - 1
                    else:
                        ner_id = self.ner_list.index(ne)

                    if ent_id not in self.ent_ids_to_ner_ids:
                        self.ent_ids_to_ner_ids[ent_id] = set([])
                    self.ent_ids_to_ner_ids[ent_id].add(ner_id)

    def get_clusters(self):
        for ent_id in self.ent_ids_to_ner_ids:
            nes = self.ent_ids_to_ner_ids[ent_id]
            if len(nes) < 2:
                continue
            cluster = []
            for ne_id in nes:
                cluster.append(self.ner_list[ne_id][:2])
            self.clusters.append(cluster)

    def get_relations(self):
        for info in self.data['interactions']:
            ent_id_1 = info['participants'][0]
            ent_id_2 = info['participants'][1]
            rel_type = [info['type']]

            for ne_id_1 in self.ent_ids_to_ner_ids[ent_id_1]:
                for ne_id_2 in self.ent_ids_to_ner_ids[ent_id_2]:
                    ne_1 = list(self.ner_list[ne_id_1][:2])
                    ne_2 = list(self.ner_list[ne_id_2][:2])
                    if (ne_1 + ne_2 + rel_type) not in self.relations:
                        self.relations.append(ne_1 + ne_2 + rel_type)



class BioRelDataset(Dataset):
  def __init__(self, json_file,
                     config={}):
    self.split = config.get('split', 'train')
    self.max_nchars_word = config.get('max_nchars_per_word', 16)
    self.max_nchars_sent = config.get('max_nchars_per_sent', 512)
    self.max_nwords = config.get('max_nwords', 100)
    self.data_dir = config.get('data_dir', './prepro_data')

    if not os.path.isdir(self.data_dir):
      os.mkdir(self.data_dir)
    self.out_path = os.path.join(self.data_dir)
    self.load_data(json_file)
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
    
    if not os.path.exists(os.path.join(self.out_path, 'rel2id.json')):
      rel2id = {}
      for item in ori_data:
        for rel in item['interactions']:
          if not str(rel['label']) in rel2id:
            rel2id[str(rel['label'])] = len(rel2id)
      json.dump(rel2id, open(self.rel2id_file, 'w'), indent=4, sort_keys=True)

    if not os.path.exists(os.path.join(self.out_path, 'ner2id.json')):
      ner2id = {}
      for item in ori_data:
        for entity in item['entities']:
          if not entity['label'] in ner2id:
            ner2id[entity['label']] = len(ner2id)
      json.dump(ner2id, open(self.ner2id_file, 'w'), indent=4, sort_keys=True)
            
    char2id = json.load(open(os.path.join(self.out_path, 'char2id.json')))
    word2id = json.load(open(os.path.join(self.out_path, 'word2id.json')))
    ner2id = json.load(open(os.path.join(self.out_path, 'ner2id.json')))
    
    sen_tot = len(ori_data)
    sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_pos = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_ner = np.zeros((sen_tot, max_length), dtype = np.int64)
    sen_char = np.zeros((sen_tot, max_length, char_limit), dtype = np.int64)

    whitespaces = {}
    new_ds = []
    for i, d in enumerate(ori_data):
      s = Sentence(d)
      doc_key = d['id']
      s = Sentence(d)
      for j, token in enumerate(s.tokens):
        if j < max_length:
          w = token.text.lower()
          if w in word2id:
            sen_word[i][j] = word2id[w]
          else:
            sen_word[i][j] = word2id['UNK']
          
          for c_idx, k in enumerate(w):
            if c_idx>=char_limit:
              break
            sen_char[i,j,c_idx] = char2id.get(k, char2id['UNK'])

      for j in range(j + 1, max_length):
        sen_word[i][j] = word2id['BLANK']

      for j, ne in enumerate(s.ner_list):
        if ne[1] > max_length:
          ne[1] = max_length - 1
        sen_ner[i][ne[0]:ne[1]+1] = ner2id[ne[2]]  
        sen_pos[i][ne[0]:ne[1]+1] = j + 1 

      whitespaces[doc_key] = s.whitespaces
      entities = []
      for ent_id in sorted(s.ent_ids_to_ner_ids, key=lambda x:int(x)):
        ent = {}
        for ner_id in s.ent_ids_to_ner_ids[ent_id]:
          ne = s.ner_list[ner_id]
          if not 'label' in ent:
            ent['label'] = ne[2]
            ent['names'] = {}
          
          if not ne[2] in ent['names']:
            ent['names'][ne[2]] = {'is_mentioned': True,
                       'mentions': [[ne[0], ne[1]+1]]}  
          else:
            ent['names'][ne[2]]['mentions'].append([ne[0], ne[1]+1]) 
        entities.append(ent)
      new_d = {'id': d['id'],
         'text': s.sentence,
         'entities': entities,
         'interactions': d['interactions']} 
      new_ds.append(new_d)
    
    json.dump(new_ds, open(os.path.join(self.out_path, self.split+'.json'), 'w'), indent=4, sort_keys=True)
    np.save(os.path.join(self.out_path, self.split+'_word.npy'), sen_word)
    np.save(os.path.join(self.out_path, self.split+'_pos.npy'), sen_pos)
    np.save(os.path.join(self.out_path, self.split+'_ner.npy'), sen_ner)
    np.save(os.path.join(self.out_path, self.split+'_char.npy'), sen_char)

  
  # def get_batch(self):
  def __getitem__(self, idx):
    return torch.LongTensor(self.word_idxs),\
           torch.LongTensor(self.char_idxs),\
           torch.LongTensor(self.ner_idxs),\
           torch.LongTensor(self.pos_idxs),\
           torch.LongTensor(self.word_lens),\
           torch.LongTensor(self.char_lens)
  
if __name__ == '__main__':
  train_set = BioRelDataset('data/1.0alpha7.train.json', {'split': 'train'})
  dev_set = BioRelDataset('data/1.0alpha7.dev.json', {'split': 'dev'})        
