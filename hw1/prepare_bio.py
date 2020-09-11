
def prepare_bio(data_file, out_file): 
  '''Convert the data into proper BIO format for entity recognition'''
  # TODO BIOES format
  tagged_sents = [] 
  with open(data_file, 'r') as f:
    data_dict = json.load(f)
  
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
      for i_ann, ann in annotations: # Check if the current token is at the begining, inside or outside an entity  
        if not label in tag_to_idx:
          tag_to_idx[label] = n_tags
          n_tags += 1               
        
        if i_pos == ann['start']:
          found = 1
          label = ','.join(ann['labels'])
          tagged_sent.append('{}/{}_B'.format(token, label))
          break
        elif i_pos > ann['start'] and i_pos < ann['end']:
          found = 1
          label = ','.join(ann['labels'])
          tagged_sent.append('{}/{}_I'.format(token, label))
          break 
      if not found:
        tagged_sent.append('{}/{}_O'.format(token, label))
    tagged_sents.append(' '.join(tagged_sent))


  # Save as a csv file
  with open('{}.csv'.format(out_file), 'w') as f_csv,
       open('{}_word_to_idx.json', 'w') as f_w,
       open('{}_tag_to_idx.json', 'w') as f_t:
    f_csv.write('\n'.join(tagged_sents))
    json.dump(f_w, indent=4, sort_keys=True)
    json.dump(f_t, indent=4, sort_keys=True)

    

