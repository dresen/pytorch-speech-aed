import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class LuongAttentionDecoderRNN(nn.Module):
    """Decoder RNN class designed to take input from an attention layer
    Arguments:
      nn {nn.Module} -- Base class to inherit from
    Returns:
      LuongAttentionDecoderRNN -- Decoder RNN network
    """
    def __init__(self, attnModel, input_size, hidden_size, output_size, nlayers=1, dropout=0.1):
        super(LuongAttentionDecoderRNN, self).__init__()
        self.attn_mdl = attnModel
        self.isz = input_size
        self.hsz = hidden_size
        self.osz = output_size
        self.nlayers = nlayers
        self.dropout = dropout

        # Layers
        self.gru = nn.GRU(input_size, hidden_size, nlayers, 
                          dropout=(0 if nlayers == 1 else dropout))
        # Calculate attention weights with a specified model
        self.attn = Attention(attnModel, hidden_size)
        # Input is concatenated encoder output and context vector
        # Both have the same dimensions as self.hsz
        # Reduce dimensions from 2*self.hsz to self.hsz
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        # Reduce dimensions from self.hsz to the output size
        # (most likely a reduction in speech)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, enc_output):
        """Pass a single "word" through the decoder nnet,
        compute attention weights and softmax
        
        Arguments:
            input_step {Tensor} -- Current Tensor slice
            last_hidden {Tensor} -- Last hidden encoder state
            enc_output {Tensor} -- Encoder output
        
        Returns:
            Tuple(Tensor, Tensor) -- Softmax distribution, decoder hidden state
        """
        # We run this method one frame at a time, so input_step is 
        # a speech frame like MFCC
        # Forward pass 
        rnn_output, hidden = self.gru(input_step, last_hidden)
        # compute atttention weights for the current steo
        attn_weights = self.attn(rnn_output, enc_output)
        # compute the new "weighted sum" context vector
        cntx = attn_weights.bmm(enc_output.transpose(0,1))
        # Concatenate rnn output and context vector
        rnn_output = rnn_output.squeeze(0)
        cntx = cntx.squeeze(1)
        concat_input = torch.cat((rnn_output,cntx), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next target word with the concatenated vector
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden