import torch
import logging
import os
import json
import time
import argparse

from data import FetDataset
from model import LstmFet
from model_joint_embed import JointEmbedLstmFet
from util import (load_word_embed,
                  load_label_embed,
                  get_word_vocab,
                  get_label_vocab,
                  calculate_macro_fscore)

def print_result(rst, vocab, mention_ids):
    rev_vocab = {i: s for s, i in vocab.items()}
    for sent_rst, mention_id in zip(rst, mention_ids):
        labels = [rev_vocab[i] for i, v in enumerate(sent_rst) if v == 1]
        print(mention_id, ', '.join(labels))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp_dir', '-e', type=str, help='Experimental directory')
parser.add_argument('--device', choices={'gpu', 'cpu'}, default='gpu', help='Use CPU or GPU')
parser.add_argument('--embedding_type', choices={'cbow', 'multilingual-cbow'}, default='cbow', help='Embedding type')
parser.add_argument('--downsample_size', type=int, default=-1, help='Size of the downsampled dataset. -1 if using the whole dataset')
parser.add_argument('--dataset', choices={'wikipedia', 'wikipedia+kbp'}, help='Dataset')
parser.add_argument('--model_type', choices={'lstm', 'lstm_joint_embed'}, default='lstm', help='Model architecture used')
args = parser.parse_args()
if not os.path.isdir('exp'):
    os.mkdir('exp')
if not os.path.isdir(args.exp_dir):
    os.mkdir(args.exp_dir)
with open('{}/args.txt'.format(args.exp_dir), 'w') as f:
    f.write(str(args))

gpu = (args.device == 'gpu')

batch_size = 250 # XXX
# Because FET datasets are usually large (1m+ sentences), it is infeasible to 
# load the whole dataset into memory. We read the dataset in a streaming way.
buffer_size = 1000 * 2000

if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isfile('data/{}_path.json'.format(args.dataset)):
  if args.dataset == 'wikipedia':
    root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/'
    path = {'root': root,
            'train': '{}/en.train.ds.json'.format(root),
            'dev': '{}/en.dev.ds.json'.format(root),
            'test': '{}/en.test.ds.json'.format(root)}
  elif args.dataset == 'wikipedia+kbp':
    root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/'
    path = {'root': root,
            'train': '{}/en.train.ds.json:{}/kbp2019.json'.format(root, root),
            'dev': '{}/en.dev.ds.json'.format(root),
            'test': '{}/en.test.ds.json'.format(root)}
  with open('data/{}_path.json'.format(args.dataset), 'w') as f:
    json.dump(path, f, indent=4, sort_keys=True)

else:
  with open('data/{}_path.json'.format(args.dataset), 'r') as f:
    path = json.load(f)
 
downsample_size = args.downsample_size
root = path['root']
train_file = path['train'] 
dev_file = path['dev'] 
test_file = path['test'] 

if downsample_size > 0:
  if ':' in train_file:
      with open('{}/train_subset_{}.json'.format(root, downsample_size), 'w') as f_subtr:
          for cur_train_file in train_file.split(':'):
              with open(cur_train_file) as f_tr:
                  for ex, line in enumerate(f_tr):
                      if ex > downsample_size:
                          break
                      print('Loading sentence {}'.format(ex))
                      f_subtr.write(line)
  else:
    with open(train_file, 'r') as f_tr,\
         open('{}/train_subset_{}.json'.format(root, downsample_size), 'w') as f_subtr:
        for ex, line in enumerate(f_tr):
            if ex > downsample_size:
                break
            print('\rLoading sentence {}'.format(ex), end='')
            train_dict = json.loads(line)
            f_subtr.write('{}\n'.format(json.dumps(train_dict)))
  train_file = '{}/train_subset_{}.json'.format(root, downsample_size)
  dev_file = '{}/train_subset_{}.json'.format(root, downsample_size)
  test_file = '{}/train_subset_{}.json'.format(root, downsample_size)
elif ':' in train_file:
    with open('{}/train_combined_{}.json'.format(root, args.dataset), 'w') as f_trc:
        for cur_train_file in train_file.split(':'):
            with open(cur_train_file) as f_tr:
                for ex, line in enumerate(f_tr):
                    print('\rLoading sentence {} in {}'.format(ex, cur_train_file), end='')
                    f_trc.write(line)
    train_file = '{}/train_combined_{}.json'.format(root, args.dataset)
if args.embedding_type == 'cbow':
    embed_file = '{}/cc.en.300.vec'.format(root) # enwiki.cbow.100d.case.txt
elif args.embedding_type == 'multilingual-cbow':
    embed_file = '{}/wiki.multi.en.vec'.format(root)
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
if not os.path.isfile('{}/vocabs.json'.format(args.exp_dir)):
  word_vocab_in_data = get_word_vocab(train_file, dev_file, test_file)
else:
  with open('{}/vocabs.json'.format(args.exp_dir), 'r') as f:
    vocabs = json.load(f)
    label_vocab = vocabs['label']
    label_num = len(label_vocab)
    word_vocab_in_data = vocabs['word']
    print(len('Total word vocabs in data={}'.format(word_vocab_in_data)))
 
word_embed, word_vocab = load_word_embed(embed_file,
                                embed_dim,
                                vocab_in_data=word_vocab_in_data,
                                skip_first=True)
print('Finish loading word embedding in {} s'.format(time.time()-begin_time))

# Scan the whole dateset to get the label set. This step may take a long 
# time. You can save the label vocab to avoid scanning the dataset 
# repeatedly.
print('Collect fine-grained entity labels')
begin_time = time.time()
if not os.path.isfile('{}/vocabs.json'.format(args.exp_dir)):
  label_vocab = get_label_vocab(train_file, dev_file, test_file)
  label_num = len(label_vocab)
  vocabs = {'word': word_vocab_in_data, 'label': label_vocab}
  with open('{}/vocabs.json'.format(args.exp_dir), 'w') as f:
    json.dump(vocabs, f, sort_keys=True, indent=4)
vocabs['word'] = word_vocab # XXX Overwrite the word indices to be consistent with the embedding matrix; Need to find a better way to handle words without embeddings
print('Number of fine-grained entity labels = {}'.format(len(vocabs['label'])))
print('Finish collecting fine-grained entity labels in {} s'.format(time.time()-begin_time))


# Build the model
print('Building the model')
if args.model_type == 'lstm':
  linear = torch.nn.Linear(embed_dim * 2, label_num)
  lstm = torch.nn.LSTM(embed_dim, embed_dim, batch_first=True)
  model = LstmFet(word_embed, lstm, linear, embed_dropout, lstm_dropout)
elif args.model_type == 'lstm_joint_embed':
  output_linear = torch.nn.Linear(embed_dim * 2, label_num)
  hidden_linear = torch.nn.Linear(embed_dim * 2, embed_dim)
  lstm = torch.nn.LSTM(embed_dim, embed_dim, batch_first=True)
  label_embed_matrix = load_label_embed(label_vocab, word_vocab, word_embed)
  if gpu:
      label_embed_matrix = label_embed_matrix.cuda()
  model = JointEmbedLstmFet(word_embed, lstm, hidden_linear, output_linear, label_embed_matrix, embed_dropout, lstm_dropout)

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

            with open('{}/dev_results_{}.json'.format(args.exp_dir, epoch), 'w') as f:
                json.dump(dev_results, f, indent=4, sort_keys=True)  
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
            with open('{}/test_results_{}.json'.format(args.exp_dir, epoch), 'w') as f:
                json.dump(test_results, f, indent=4, sort_keys=True)
            print('Test set (Macro): P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
                precision, recall, fscore))
            
            if best_dev:
                best_test_score = fscore
        
    torch.save(model.state_dict(), '{}/model.pth'.format(args.exp_dir, epoch))
    print()
    print('Loss: {:.4f}'.format(sum(losses) / len(losses)))

print('Best macro F-score (dev): {:2f}'.format(best_dev_score))
print('Best macro F-score (test): {:2f}'.format(best_test_score))
        
    
