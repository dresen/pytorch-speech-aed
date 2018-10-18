import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Attention layer with supprt for different attention mechanisms
    
    Arguments:
        nn {nn.Module} -- Module to we inherit from
        
    Returns:
        Attention -- [Attention object]
    """
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.Tensor(self.hidden_size))

    def dot_score(self, hidden, enc_output):
        return torch.sum(hidden * enc_output, dim=2)

    def scaled_dot_score(self, hidden, enc_output, lengths):
        return torch.div(torch.sum(hidden * enc_output, dim=2), torch.sqrt(length))

    def general_score(self, hidden, enc_output):
        energy = self.attn(enc_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, enc_output):
        energy = self.attn(torch.cat((hidden.expand(enc_output.size(0), -1, -1), enc_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, enc_output, lengths=False):
        """Forward pass throuch the attention layer"""
        if self.method == 'general':
            attn_energies = self.general_score(hidden, enc_output)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, enc_output)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, enc_output)
        elif self.method == 'scaled':
            attn_energies = self.dot_score(hidden, enc_output, lengths)

        # Switch max len and batch size dimensions
        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)