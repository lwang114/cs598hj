import json
import os
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
# from nltk.tree import Tree
# from nltk.parse import stanford
# from nltk.tokenize import sent_tokenize 
from copy import deepcopy

def extract_dependency_parse(data_file, out_file, dep_parser_path='/Users/liming/nltk_data/stanford-parser-full-2018-10-17/edu/stanford/nlp/models/parser/nndep/english_UD.gz'):
  # Load the dependency parser
  dep_parser = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz") # stanford.StanfordDependencyParser(path_to_models_jar=dep_parser_path)
  
  with open(data_file, 'r') as f_in,\
       open('{}.json'.format(out_file), 'w') as f_out:  
      for ex, line in enumerate(f_in):
        # if ex > 30: # XXX
        #   break 
        print('\rExample {}'.format(ex), end='')
        data_dict = json.loads(line)
        sent = data_dict['tokens']

        # Parse the sentence
        sent_len = len(sent)
        parsed_sent = dep_parser.predict(' '.join(sent))
        predicted_labels = parsed_sent['predicted_dependencies']
        predicted_heads = parsed_sent['predicted_heads']
        dep_parse_dict = {'predicted_dependencies': predicted_labels,
                          'predicted_heads': predicted_heads}

        # Save the parse info into the data dict
        data_dict['dep_parse'] = dep_parse_dict
        f_out.write('{}\n'.format(json.dumps(data_dict)))
        

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
  path_file = 'data/path.json'
  root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/'

  if not os.path.isdir('data'):
    os.mkdir('data')
  if not os.path.isfile(path_file):

    path = {'root': root,
            'train': '{}/kbp2019.json'.format(root),
            'dev': '{}/kbp2019.json'.format(root), # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.dev.json',
            'test': '{}/kbp2019.json'.format(root)} # '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.test.json'}
    with open(path_file, 'w') as f:
      json.dump(path, f, indent=4, sort_keys=True)
  else:
    with open(path_file, 'r') as f:
      path = json.load(f)
  
  extract_dependency_parse(path['train'], out_file='{}_dep_parsed.json'.format(path['train'].split('.json')[0]))
  extract_dependency_parse(path['dev'], out_file='{}_dep_parsed.json'.format(path['dev'].split('.json')[0]))
  extract_dependency_parse(path['test'], out_file='{}_dep_parsed.json'.format(path['test'].split('.json')[0]))
