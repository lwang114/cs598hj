import json
import os

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
  elif not os.path.isfile('data/path.json'):
    path = {'train': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.train.json',
            'dev': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.dev.json',
            'test': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/en.test.json'}
    with open('data/path.json', 'w') as f:
      json.dump(path, f, indent=4, sort_keys=True)
  else:
    with open('data/path.json', 'r') as f:
      path = json.load(f)

  downsample_size = 10
  with open(path['dev'], 'r') as f_tr,\
       open(path['test'], 'r') as f_tx:
    train_dict = [json.loads(line) for line in f_tr]
    test_dict = [json.loads(line) for line in f_tx]
  
  if downsample_size > 0:
    with open('data/train_subset.json', 'w') as f_subtr,\
         open('data/test_subset.json', 'w') as f_subtx:
      for cur_train_dict in train_dict[:downsample_size]:
        f_subtr.write('{}\n'.format(json.dumps(cur_train_dict)))

      for cur_test_dict in test_dict[:downsample_size]:
        f_subtx.write('{}\n'.format(json.dumps(cur_test_dict)))
    path['train'] = 'data/train_subset.json'
    path['test'] = 'data/test_subset.json' 
  
  prepare_iob(path['train'], 'data/train_iob')
  prepare_iob(path['test'], 'data/test_iob') 
