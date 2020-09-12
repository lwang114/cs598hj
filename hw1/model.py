import torch
import torch.nn as nn
import torch.nn.utils.rnn as R
from typing import Tuple


class LstmFet(nn.Module):
    def __init__(self,
                 word_embed: nn.Embedding,
                 lstm: nn.LSTM,
                 output_linear: nn.Linear,
                 word_embed_dropout: float = 0,
                 lstm_dropout: float = 0):
        super().__init__()
        
        self.word_embed = word_embed
        self.lstm = lstm
        self.output_linear = output_linear
        self.word_embed_dropout = nn.Dropout(word_embed_dropout)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.criterion = nn.MultiLabelSoftMarginLoss()
    
    def forward_nn(self,
                   inputs: torch.Tensor,
                   mention_mask: torch.Tensor,
                   context_mask: torch.Tensor,
                   seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Word index tensor for the input batch.
            mention_mask (torch.Tensor): A mention mask with the same size of
              `inputs`.
            context_mask (torch.Tensor): A context mask with the same size of 
              `inputs`.
            seq_lens (torch.Tensor): A vector of sequence lengths.
            
            If a sequence has 6 tokens, where the 2nd token is a mention, and 
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]
            
        Returns:
            torch.Tensor: label scores. A NxM matrix where N is the batch size 
              and M is the number of labels.
        """
        inputs_embed = self.word_embed(inputs)
        inputs_embed = self.word_embed_dropout(inputs_embed)
        
        lstm_in = R.pack_padded_sequence(inputs_embed,
                                         seq_lens,
                                         batch_first=True)
        
        lstm_out = self.lstm(lstm_in)[0]
        lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Average mention embedding
        mention_mask = ((1 - mention_mask) * -1e14).softmax(-1).unsqueeze(-1)
        mention_repr = (lstm_out * mention_mask).sum(1)
        
        # Average context embedding
        context_mask = ((1 - context_mask) * -1e14).softmax(-1).unsqueeze(-1)
        context_repr = (lstm_out * context_mask).sum(1)
        
        # Concatenate mention and context representations
        combine_repr = torch.cat([mention_repr, context_repr], dim=1)
        
        # Linear classifier
        scores = self.output_linear(combine_repr)
        
        return scores
    
    def forward(self,
                inputs: torch.Tensor,
                mention_mask: torch.Tensor,
                context_mask: torch.Tensor,
                labels: torch.Tensor,
                seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs (torch.Tensor): Word index tensor for the input batch.
            mention_mask (torch.Tensor): A mention mask with the same size of
            `inputs`.
            context_mask (torch.Tensor): A context mask with the same size of 
            `inputs`.
            labels (torch.Tensor): A tensor of label vectors. The label vector 
              for each mention is a binary vector where each value indicates 
              whether the corresponding label is assigned to the mention or not.
            seq_lens (torch.Tensor): A vector of sequence lengths.
            
            If a sequence has 6 tokens, where the 2nd token is a mention, and 
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The first element is the loss.
              The second element are label scores, a NxM matrix where N is 
              the batch size and M is the number of labels.
        """
        scores = self.forward_nn(inputs, mention_mask, context_mask, seq_lens)
        loss = self.criterion(scores, labels)
        return loss, scores
    
    def predict(self,
                inputs: torch.Tensor,
                mention_mask: torch.Tensor,
                context_mask: torch.Tensor,
                seq_lens: torch.Tensor,
                predict_top: bool = True) -> torch.Tensor:
        """Predict fine-grained entity types of a batch of mentions.

        Args:
            inputs (torch.Tensor): word index tensor for the input batch.
            mention_mask (torch.Tensor): a mention mask with the same size of
              `inputs`.
            context_mask (torch.Tensor): a context mask with the same size of 
              `inputs`.
            seq_lens (torch.Tensor): A vector of sequence lengths.
            predict_top (bool, optional): if True, a label will be predicted
              even if its score (after sigmoid) is smaller than 0.5. Defaults 
              to True.
            
            If a sequence has 6 tokens, where the 2nd token is a mention, and 
            the longest sequence in the current batch has 8 tokens, the mention
            mask and context mask of this sequence are:
            - mention mask: [0, 1, 0, 0, 0, 0, 0, 0]
            - context mask: [1, 1, 1, 1, 1, 1, 0, 0]
            
        Returns:
            torch.Tensor: prediction result. A NxM matrix where N is the batch 
              size and M is the number of labels. Label j is predicted for the
              i-th mention if the i,j element is 1.
        """
        self.eval()
        scores = self.forward_nn(inputs, mention_mask, context_mask, seq_lens)
        
        predictions = (scores.sigmoid() > .5).int()
        
        if predict_top:
            _, highest = scores.max(dim=1)
            highest = highest.int().tolist()
            for i, h in enumerate(highest):
                predictions[i][h] = 1
        
        self.train()
        
        return predictions