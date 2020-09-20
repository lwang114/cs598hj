import json
import os
from nltk.tree import Tree
from nltk.parse import stanford
from nltk.tokenize import sent_tokenize 
from copy import deepcopy


def extract_dependency_parse(data_file, out_file, dep_parser_path='/Users/liming/nltk_data/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/parser/nndep/english_UD.gz'):
  # Load the dependency parser
  dep_parser = stanford.StanfordDependencyParser(path_to_models_jar=dep_parser_path)
  
  with open(data_file, 'r') as f_in,\
       open('{}_parse_trees.json'.format(out_file), 'w') as f_out_tree,\
       open('{}_adjacency_matrices.json'.format(out_file), 'w') as f_out_adj:  
      for ex, line in enumerate(f_in):
        if ex > 30: # XXX
          break 
        data_dict = json.loads(line)
        sent = sent_tokenize(' '.join(data_dict['tokens']))
        
        sent_len = len(sent)
        # Parse the sentence
        parsed_sent = dep_parser.raw_parse_sents(sent)
        tree = Tree.fromstring(str(parsed_sent))
        queue = [tree]
        A = np.zeros((sent_len, sent_len))
        # Traverse the tree to generate the adjacency matrix
        while len(queue) != 0: # TODO
          cur_tree = queue.pop(0)
          for child in cur_tree:
            if isinstance(child, Tree):
              queue.append(child)
            else:
              A[cur_tree.label(), child] = 1.
            
        f_out_adj.write('{}\n'.format(json.dumps({'adjacency_matrix': A.tolist()}, indent=4, sort_keys=True)))
        f_out_tree.write('{}\n'.format(tree.pprint()))
        

def prepare_iob(data_file, out_file): 
  '''Convert the data into proper BIO format for entity recognition'''
  # TODO BIOES format
  tagged_sents = [] 
  with open(data_file, 'r') as f:
    data_dict = [json.loads(line) for line in f]
  
  word_to_idx = {}  
  tag_to_idx = {}
  n_words = 0
  n_tags = 0
  for datum_dict in data_dict:
    tokens = datum_dict['tokens']
    annotations = datum_dict['annotations']
    tagged_sent = []
    for i_pos, token in enumerate(tokens): # Loop over annotations of each entity mention
      if not token in word_to_idx:
        word_to_idx[token] = n_words
        n_words += 1

      found = 0
      for i_ann, ann in enumerate(annotations): # Check if the current token is at the begining, inside or outside an entity  
        label = ','.join(sorted(ann['labels']))
        if not label in tag_to_idx:
          tag_to_idx[label] = n_tags
          n_tags += 1               
        
        if i_pos == ann['start']:
          found = 1
          tagged_sent.append('{}/{}_B'.format(token, label))
          break
        elif i_pos > ann['start'] and i_pos < ann['end']:
          found = 1
          tagged_sent.append('{}/{}_I'.format(token, label))
          break 
      if not found:
        tagged_sent.append('{}/O'.format(token))
    tagged_sents.append(' '.join(tagged_sent))


  # Save as a csv file
  with open('{}.csv'.format(out_file), 'w') as f_csv,\
       open('{}_word_to_idx.json'.format(out_file), 'w') as f_w,\
       open('{}_tag_to_idx.json'.format(out_file), 'w') as f_t:
    f_csv.write('\n'.join(tagged_sents))
    json.dump(word_to_idx, f_w, indent=4, sort_keys=True)
    json.dump(tag_to_idx, f_t, indent=4, sort_keys=True)

if __name__ == '__main__':
  if not os.path.isdir('data'):
    os.mkdir('data')
  if not os.path.isfile('data/path.json'):
    path = {'train': 'kbp2019.json',
            'dev': 'kbp2019.json', # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.dev.json',
            'test': 'kbp2019.json'} # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.test.json'}
    with open('data/path.json', 'w') as f:
      json.dump(path, f, indent=4, sort_keys=True)
  else:
    with open('data/path.json', 'r') as f:
      path = json.load(f)
  
  extract_dependency_parse(path['train'], out_file='kbp')
