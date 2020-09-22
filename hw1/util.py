import json
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Tuple
import numpy as np
from copy import deepcopy

def load_word_embed(path: str,
                    dimension: int,
                    *,
                    skip_first: bool = False,
                    freeze: bool = False,
                    sep: str = ' ',
                    vocab_in_data: Dict[str, int] = None
                    ) -> Tuple[nn.Embedding, Dict[str, int]]:
    """Load pre-trained word embeddings from file.

    Args:
        path (str): Path to the word embedding file.
        skip_first (bool, optional): Skip the first line. Defaults to False.
        freeze (bool, optional): Freeze embedding weights. Defaults to False.
        
    Returns:
        Tuple[nn.Embedding, Dict[str, int]]: The first element is an Embedding
        object. The second element is a word vocab, where the keys are words and
        values are word indices.
    """
    vocab = {'$$$UNK$$$': 0}
    
    if ':' in path: # Combine embeddings
        n_paths = len(path.split(':'))
        embed_matrix = [[[0.0] * dimension for _ in range(n_paths)]]
        for i_path, subpath in enumerate(path.split(':')):
            with open(subpath) as r:
                if skip_first:
                    r.readline()
                for line in r:
                    # if len(vocab) > 100: # XXX
                    #     break 
                    segments = line.rstrip('\n').rstrip(' ').split(sep)
                    word = segments[0]
                    if not word in vocab_in_data:
                        continue
                    embed = [float(x) for x in segments[1:]]
                    if not word in vocab:
                        vocab[word] = len(vocab)    
                        embed_matrix.append([embed for _ in range(n_paths)])
                    else:
                        embed_matrix[vocab[word]][i_path] = deepcopy(embed)
                    print('\rFound embedding {} for {} from path {}'.format(len(vocab), word, subpath), end='')
        embed_matrix = torch.FloatTensor(embed_matrix)
        embed_matrix = embed_matrix.reshape(embed_matrix.size(0), -1)
        print('embed_matrix.size(): {}'.format(embed_matrix.size()))
    else:
        embed_matrix = [[0.0] * dimension]
        with open(path) as r:
            if skip_first:
                r.readline()
            for line in r:
                segments = line.rstrip('\n').rstrip(' ').split(sep)
                word = segments[0]
                if not word in vocab_in_data:
                    continue 

                # if len(vocab) > 100: # XXX
                #   break 
                print('\rEmbedding {} for {}'.format(len(vocab), word), end='')
                vocab[word] = len(vocab)
                embed = [float(x) for x in segments[1:]]
                embed_matrix.append(embed)
                print('\rLoaded %d word embeddings' % (len(embed_matrix) - 1))
            
        embed_matrix = torch.FloatTensor(embed_matrix)
    
    word_embed = nn.Embedding.from_pretrained(embed_matrix,
                                              freeze=freeze,
                                              padding_idx=0)
    return word_embed, vocab

def load_label_embed(label_vocabs: Dict[str, int], 
                     word_vocabs: Dict[str, int],
                     word_embed: nn.Embedding,
                     freeze: bool = False) -> nn.Embedding:
    """Extract label embeddings from pretrained word embeddings
    
    Args:
        label_vocabs (dict): {label: idx for idx, label in enumerate(labels)}
        word_vocabs (dict): {word: idx for idx, word in enumerate(vocabs)}
        word_embed: Embedding object storing the pretrained word embeddings
    
    Returns:
        torch.Tensor: Tensor storing the label word embeddings 
    """
    label_embed_matrix = []
    for i_label, label in enumerate(sorted(label_vocabs, key=lambda x:label_vocabs[x])):
      label_word = label[:-9].lower()
      nchars = len(label_word)
      if label_word in word_vocabs:
        label_embed_matrix.append(word_embed(torch.LongTensor([word_vocabs.get(label_word)])))
      else:
        found = False
        for i_char in range(nchars):
          if label_word[:i_char] in word_vocabs and label_word[i_char:] in word_vocabs:
            w1 = label_word[:i_char]
            w2 = label_word[i_char:]
            print('Words for label {}: {} {}'.format(i_label, label_word[:i_char], label_word[i_char:]))
            v1 = torch.LongTensor([word_vocabs.get(w1)])
            v2 = torch.LongTensor([word_vocabs.get(w2)])
            label_embed_matrix.append(torch.mean(torch.cat([word_embed(v1), word_embed(v2)]), dim=0, keepdim=True))
            found = True
            break

        if not found:
          print('Embedding not found for label {}: {}'.format(i_label, label_word))
          v = word_embed(torch.LongTensor([0]))
          label_embed_matrix.append(v)
    label_embed_matrix = torch.cat(label_embed_matrix).data
    return label_embed_matrix
    
def get_word_vocab(*paths: str) -> Dict[str, int]:
    """Generate a word vocab from data files.
    
    Args:
        paths (str): data file paths

    Returns:
        Dict[str, int]: A word vocab where keys are labels and values are label
        indices
    
    """
    word_set = set()
            
    for path in paths:
      if ':' in path:
        for subpath in path.split(':'):
            with open(subpath) as r:
                for line in r:
                    instance = json.loads(line)
                    tokens = instance['tokens']
                    tokens += [token.lower() for token in tokens] 
                    word_set.update(tokens)
      else:
          with open(path) as r:
              for line in r:
                  instance = json.loads(line)
                  tokens = instance['tokens']
                  tokens += [token.lower() for token in tokens] 
                  word_set.update(tokens)

    return {word: idx for idx, word in enumerate(word_set)}

def get_label_vocab(*paths: str) -> Dict[str, int]:
    """Generate a label vocab from data files.

    Args:
        paths (str): data file paths.
    
    Returns:
        Dict[str, int]: A label vocab where keys are labels and values are label
        indices.
    """
    label_set = set()
    for path in paths:
        if ':' in path: # Combine data from multiple files
          for subpath in path.split(':'):
            with open(subpath) as r:            
                for line in r:
                  instance = json.loads(line)
                  for annotation in instance['annotations']:
                      label_set.update(annotation['labels'])

        else:
          with open(path) as r:
              for line in r:
                  instance = json.loads(line)
                  for annotation in instance['annotations']:
                      label_set.update(annotation['labels'])
    return {label: idx for idx, label in enumerate(label_set)}

def calculate_macro_fscore(golds: List[List[int]],
                           preds: List[List[int]]
                           ) -> Tuple[float, float, float]:
    """Calculate Macro F-score.

    Args:
        golds (List[List[int]]): Ground truth. The j-th element in the i-th
        list indicates whether the j-th label is associated with the i-th
        entity or not. If it is 1, the entity is annotated with the j-th
        label. If it is 0, the j-th label is not assigned to the entity.
        preds (List[List[int]]): Prediction. The j-th element in the i-th
        list indicates whether the j-th label is predicted for the i-th
        entity or not.

    Returns:
        Tuple[float, float, float]: Precision, recall, and F-score.
    """
    total_gold_num = total_pred_num = 0
    precision = recall = 0
    for gold, pred in zip(golds, preds):
        gold_num = sum(gold)
        pred_num = sum(pred)
        total_gold_num += (1 if gold_num > 0 else 0)
        total_pred_num += (1 if pred_num > 0 else 0)
        overlap = sum([i and j for i, j in zip(gold, pred)])
        precision += (0 if pred_num == 0 else overlap / pred_num)
        recall += (0 if gold_num == 0 else overlap / gold_num)
    precision = precision / total_pred_num if total_pred_num else 0
    recall = recall / total_gold_num if total_gold_num else 0
    fscore = 0 if precision + recall == 0 else \
        2.0 * (precision * recall) / (precision + recall)

    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_micro_fscore(golds: List[List[int]],
                           preds: List[List[int]]) -> Tuple[float, float, float]:
    """Calculate Micro F-score"""
    overlap = 0
    total_gold_num = total_pred_num = 0
    precision = recall = 0
    for gold, pred in zip(golds, preds):
        total_gold_num += sum(gold)
        total_pred_num += sum(pred)
        overlap += sum([i and j for i, j in zip(gold, pred)])
    precision = overlap / total_pred_num if total_pred_num else 0
    recall = overlap / total_gold_num if total_gold_num else 0
    fscore = 0 if precision + recall == 0 else \
             2.0 * (precision * recall) / (precision + recall)
    return precision * 100.0, recall * 100.0, fscore * 100.0

def calculate_accuracy(golds: List[List[int]],
                       preds: List[List[int]]) -> float:
    """Calculate Accuracy"""
    correct = 0
    total_num = len(golds)
    for gold, pred in zip(golds, preds):
        correct += (gold == pred)

    accuracy = correct / total_num
    return accuracy

def print_error_examples(in_fn, pred_fn, out_fn, label_vocab_fn, max_n_mentions=200):
  """Save 200 error examples randomly chosen from the dataset"""
  with open(pred_fn, 'r') as pred_f,\
       open(label_vocab_fn, 'r') as vocab_f:
    pred_dict = json.load(pred_f)
    label_vocab_dict = json.load(vocab_f)['label']
    label_vocabs = sorted(label_vocab_dict, key=lambda x:label_vocab_dict[x])

    preds = pred_dict['pred']
    golds = pred_dict['gold']
    mention_ids = pred_dict['id']
    
  with open(in_fn, 'r') as in_f,\
       open(out_fn, 'w') as out_f:
    i_err = 0
    for line in in_f:
      sent_dict = json.loads(line)
      sent = sent_dict['tokens']
      for mention_dict in sent_dict['annotations']:
        cur_mention_id = mention_dict['mention_id']
        for i_mention, mention_id in enumerate(mention_ids):
            if mention_id == cur_mention_id:
                break
        mention = mention_dict['mention']
        start, end = mention_dict['start'], mention_dict['end']
        labels = mention_dict['labels']
        pred = preds[i_mention]
        gold = golds[i_mention]
        gold_label = [label_vocabs[k] for k, i in enumerate(gold) if i]
        pred_label = [label_vocabs[k] for k, i in enumerate(pred) if i]
        # if i_err < 10:
        #     print(sent, mention_dict['mention'], mention_dict['labels'], gold_label)
        
        if gold == pred:
            continue
        i_err += 1
        out_f.write('{}\nMention: {} {} {}\nLabel: {}\nGold: {}\nPred: {}\n\n'.format(' '.join(sent),\
                                                                                      mention, start, end, ' '.join(labels), ' '.join(gold_label), ' '.join(pred_label)))
        if i_err >= max_n_mentions:
            break
      if i_err >= max_n_mentions:
        break
  print('Number of mentions {}, number of errors {}'.format(len(preds), i_err))
    
def multi_max_margin_rank_loss(scores: torch.Tensor,
                         labels: torch.Tensor,
                         margin: float = 1.) -> torch.Tensor:
    """Compute the multi-margin rank loss
    
    Args:
        scores (Tensor): B x K tensor storing the class similarity scores
        labels (Tensor): B x K tensor storing the binary class label vecotrs
    
    Returns:
        torch.Tensor: Scalar storing the average max margin rank loss   
    """
    B = scores.size(0)
    loss = torch.zeros(1, device=scores.device, requires_grad=True)
    score_diffs = (scores.unsqueeze(1) - scores.unsqueeze(2))
    thres = torch.FloatTensor(np.zeros((B, 1, 1))).to(scores.device)
    score_diffs_thresholded = torch.max(score_diffs + margin, thres)
    loss = loss + torch.sum(torch.sum(score_diffs_thresholded * labels.unsqueeze(2), axis=-1) * (1 - labels).unsqueeze(1))
    return loss / B

if __name__ == '__main__':
    import os
    data_root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/cs598hj/hw1/data/'
    exp_root = 'exp/'
    in_fn = '{}/en.test.ds.json'.format(data_root)
    i_epoch = 14
    for exp_dir in os.listdir(exp_root):
        print(exp_dir)
        exp_dir = '{}/{}'.format(exp_root, exp_dir)
        pred_fn = '{}/test_results_{}.json'.format(exp_dir, i_epoch)
        if not os.path.isfile(pred_fn):
            continue
        label_vocab_fn = '{}/vocabs.json'.format(exp_dir)
        print_error_examples(in_fn, pred_fn, '{}/error_examples.txt'.format(exp_dir), label_vocab_fn)
        with open(pred_fn, 'r') as f:
            pred_dict = json.load(f)
            golds = pred_dict['gold']
            preds = pred_dict['pred']
            prec, rec, f1 = calculate_micro_fscore(golds, preds)
            acc = calculate_accuracy(golds, preds)
            print('Micro precision {:2f}, recall {:2f}, F1 score {:2f}'.format(prec, rec, f1))
            print('Accuracy {:2f}\n'.format(acc))
