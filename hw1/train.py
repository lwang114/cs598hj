import torch
import logging
import os
import json
import time

from data import FetDataset
from model import LstmFet
from util import (load_word_embed,
                  get_word_vocab,
                  get_label_vocab,
                  calculate_macro_fscore)

def print_result(rst, vocab, mention_ids):
    rev_vocab = {i: s for s, i in vocab.items()}
    for sent_rst, mention_id in zip(rst, mention_ids):
        labels = [rev_vocab[i] for i, v in enumerate(sent_rst) if v == 1]
        print(mention_id, ', '.join(labels))

gpu = True

batch_size = 1000
# Because FET datasets are usually large (1m+ sentences), it is infeasible to 
# load the whole dataset into memory. We read the dataset in a streaming way.
buffer_size = 1000 * 2000

if not os.path.isdir('data'):
  os.mkdir('data')
elif not os.path.isfile('data/path.json'):
  root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/'
  path = {'root': root,
          'train': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/en.train.ds.json',
          'dev': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/en.dev.ds.json',
          'test': '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/en.test.ds.json'}
  with open('data/path.json', 'w') as f:
    json.dump(path, f, indent=4, sort_keys=True)
else:
  with open('data/path.json', 'r') as f:
    path = json.load(f)
 
downsample_size = 100
root = path['root']
if downsample_size > 0:
  with open(path['train'], 'r') as f_tr,\
       open('{}/train_subset_{}.json'.format(root, downsample_size), 'w') as f_subtr:
    for ex, line in enumerate(f_tr):
      if ex > downsample_size:
        break
      print('Loading sentence {}'.format(ex))
      train_dict = json.loads(line)
      f_subtr.write('{}\n'.format(json.dumps(train_dict)))
  path['train'] = '{}/train_subset_{}.json'.format(root, downsample_size)
  path['dev'] = '{}/train_subset_{}.json'.format(root, downsample_size)
  path['test'] = '{}/train_subset_{}.json'.format(root, downsample_size)

train_file = path['train'] 
dev_file = path['dev'] 
test_file = path['test'] 

# if os.path.isfile('{}/cc.en.300.indata.vec'.format(root)):
#   embed_file = '{}/cc.en.300.indata.vec'.format(root)
# else:
embed_file = '{}/cc.en.300.vec'.format(root) # enwiki.cbow.100d.case.txt
embed_dim = 300 # 100
embed_dropout = 0.5
lstm_dropout = 0.5

lr = 1e-4
weight_decay = 1e-3
max_epoch = 15

# Datasets
train_set = FetDataset(train_file)
dev_set = FetDataset(dev_file)
test_set = FetDataset(test_file)

# Load word embeddings from file
# If the word vocab is too large for your machine, you can remove words not
# appearing in the data set.
print('Loading word embeddings from %s' % embed_file)
begin_time = time.time()
if not os.path.isfile('{}/vocabs.json'.format(root)):
  word_vocab_in_data = get_word_vocab(train_file, dev_file, test_file)
else:
  with open('{}/vocabs.json'.format(root), 'r') as f:
    vocabs = json.load(f)
    label_vocab = vocabs['label']
    label_num = len(label_vocab)
    word_vocab_in_data = vocabs['word']
    print(len('Total word vocabs in data={}'.format(word_vocab_in_data)))
 
# XXX word_embed, word_vocab = load_word_embed(embed_file,
word_embed, word_vocab = load_word_embed(embed_file,
                                embed_dim,
                                vocab_in_data=word_vocab_in_data, # XXX
                                skip_first=True)
print('Finish loading word embedding in {} s'.format(time.time()-begin_time))

# Scan the whole dateset to get the label set. This step may take a long 
# time. You can save the label vocab to avoid scanning the dataset 
# repeatedly.
print('Collect fine-grained entity labels')
begin_time = time.time()
if not os.path.isfile('{}/vocabs.json'.format(root)):
  label_vocab = get_label_vocab(train_file, dev_file, test_file)
  label_num = len(label_vocab)
  vocabs = {'word': word_vocab_in_data, 'label': label_vocab}
  with open('{}/vocabs.json'.format(root), 'w') as f: # XXX
    json.dump(vocabs, f, sort_keys=True, indent=4)
print('Finish collecting fine-grained entity labels in {} s'.format(time.time()-begin_time))

# Build the model
print('Building the model')
linear = torch.nn.Linear(embed_dim * 2, label_num)
lstm = torch.nn.LSTM(embed_dim, embed_dim, batch_first=True)
model = LstmFet(word_embed, lstm, linear, embed_dropout, lstm_dropout)
if gpu:
    model.cuda()

# Optimizer: Adam with decoupled weight decay
optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad,
                                     model.parameters()),
                              lr=lr,
                              weight_decay=weight_decay)

best_dev_score = best_test_score = 0
for epoch in range(max_epoch):
    print('Epoch %d' % epoch)
    
    # Training set
    losses = []
    for idx, batch in enumerate(train_set.batches(vocabs,
                                                  batch_size,
                                                  buffer_size,
                                                  shuffle=True,
                                                  gpu=gpu)):
        print('\rBatch %d' % idx, end='')
        optimizer.zero_grad()
        
        # Unpack the batch
        (token_idxs, labels,
         mention_mask, context_mask,
         mention_ids, mentions, seq_lens) = batch

        loss, scores = model(token_idxs,
                             mention_mask,
                             context_mask,
                             labels,
                             seq_lens)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        if idx and idx % 500 == 0:
            print()
            # Dev set
            best_dev = False
            dev_results = {'gold': [], 'pred': [], 'id': []}
            for batch in dev_set.batches(vocabs,
                                        batch_size // 10,
                                        buffer_size,
                                        gpu=gpu,
                                        max_len=1000):
                # Unpack the batch
                (token_idxs, labels,
                mention_mask, context_mask,
                mention_ids, mentions, seq_lens) = batch
                
                predictions = model.predict(token_idxs,
                                            mention_mask,
                                            context_mask,
                                            seq_lens)
                dev_results['gold'].extend(labels.int().tolist())
                dev_results['pred'].extend(predictions.int().tolist())
                dev_results['id'].extend(mention_ids)
            precision, recall, fscore = calculate_macro_fscore(dev_results['gold'],
                                                            dev_results['pred'])
            print('Dev set (Macro): P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
                precision, recall, fscore))
            if fscore > best_dev_score:
                best_dev_score = fscore
                best_dev = True
            
            # Test set
            test_results = {'gold': [], 'pred': [], 'id': []}
            for batch in test_set.batches(vocabs,
                                        batch_size // 10,
                                        buffer_size,
                                        gpu=gpu,
                                        max_len=1000):
                # Unpack the batch
                (token_idxs, labels,
                mention_mask, context_mask,
                mention_ids, mentions, seq_lens) = batch
                
                predictions = model.predict(token_idxs,
                                            mention_mask,
                                            context_mask,
                                            seq_lens)
                test_results['gold'].extend(labels.int().tolist())
                test_results['pred'].extend(predictions.int().tolist())
                test_results['id'].extend(mention_ids)
            precision, recall, fscore = calculate_macro_fscore(test_results['gold'],
                                                            test_results['pred'])
            print('Test set (Macro): P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
                precision, recall, fscore))
            
            if best_dev:
                best_test_score = fscore
        
    print()
    print('Loss: {:.4f}'.format(sum(losses) / len(losses)))

print('Best macro F-score (dev): {:2.f}'.format(best_dev_score))
print('Best macro F-score (test): {:2.f}'.format(best_test_score))
        
    
