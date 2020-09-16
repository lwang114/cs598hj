import json
import torch
import torch.nn as nn
import logging
from typing import List, Dict, Tuple


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
            print('Embedding {} for {}'.format(len(vocab), word))
            vocab[word] = len(vocab)
            embed = [float(x) for x in segments[1:]]
            embed_matrix.append(embed)
    print('Loaded %d word embeddings' % (len(embed_matrix) - 1))
            
    embed_matrix = torch.FloatTensor(embed_matrix)
    
    word_embed = nn.Embedding.from_pretrained(embed_matrix,
                                              freeze=freeze,
                                              padding_idx=0)
    print(word_embed)
    return word_embed, vocab
 
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

''' TODO
def max_margin_rank_loss(scores: torch.Tensor,
                         labels: torch.Tensor,
                         margin: int = 1.):
    loss = torch.zeros(1)
    B = scores.size(0)
    for b in range(B):
      positive_indices = np.nonzeros(labels[b].cpu().numpy())[0]
      negative_indices = np.nonzeros(1. - labels[b].cpu().numpy())[0]
      for i_pos in positive_indices:
        for i_neg in negative_indices:
          loss += torch.max(scores[b][i_neg] - scores[b][i_pos] + margin, 0.)
    return loss / B
'''   
