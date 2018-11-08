"""GRU-based sequence models for CTC training
"""

import torch
import torch.nn.functional as F
from models.lstm import SequenceLinear

class BaseGru(torch.nn.Module):
    """Base class for recurrent net with GRU units that can work a
    an encoder net
    
    Arguments:
        torch {Module} -- Base inherited module from torch
    
    Returns:
        BaseGru -- A GRU-based recurrent net
    """
    def __init__(self, input_size, hidden_size, nlayers, dropout=0, bidi=False, batch_first=True):
        super(BaseGru, self).__init__()
        self.isz = input_size      # Number of features
        self.hsz = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.directions = 2 if bidi else 1
        self.batch_first=batch_first

        self.encoder = torch.nn.GRU(self.isz, self.hsz, self.nlayers, dropout=self.dropout, 
        batch_first=True, bidirectional=bidi)

    def forward(self, sequences, sequence_lengths, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths,
                                                         batch_first=self.batch_first)
        enc_out, new_hidden_state = self.encoder(packed, hidden)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out,
                                                                batch_first=True)
        if self.directions == 2:
            # Sum both directions
            unpacked = unpacked[:,:,:self.hsz] + unpacked[:,:,self.hsz:]
        return unpacked, new_hidden_state


class CTCgru(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(CTCgru, self).__init__(*args, **kwargs)
        self.isz = input_size      # Number of features
        self.osz = output_size + 1 # Remember to add "blank" to the output size if we use CTCLoss
        self.hsz = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.encoder = torch.nn.GRU(self.isz, self.hsz, dropout=self.dropout, 
        batch_first=True, bidirectional=bidi)

        self.decoder = SequenceLinear(self.hsz, self.osz)

        self.logsoftmax = torch.nn.LogSoftmax(2)


    def forward(self, sequences):
        enc_out, _ = self.encoder(sequences)
        dec_out = self.decoder(enc_out)
        return self.logsoftmax(dec_out)


class CTCgrup(BaseGru):
    """A recurrent NN designed for CTC training that packs the input to the NN in
    the forward pass. The encoder is a BaseGru object and we add a decoder a
    softmax output, and blank output node
    
    Arguments:
        BaseGru {BaseGru} -- The base class defined above
    
    Returns:
        CTCGrup -- A NN we can train with CTC
    """
    def __init__(self, input_size, output_size, hidden_size, nlayers, dropout=0, bidi=False):
        super(CTCgrup, self).__init__(input_size, hidden_size, nlayers, dropout=0, bidi=False)
        self.pack_unpack = True
        self.osz = output_size + 1
        self.decoder = SequenceLinear(self.hsz, self.osz)

        self.logsoftmax = torch.nn.LogSoftmax(2)

    def forward(self, sequences, sequence_lengths, hidden_state=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths, batch_first=True)
        enc_out, _ = self.encoder(packed, hidden_state)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        if self.directions == 2:
            # Sum both directions so we get the right dimensions
            unpacked = unpacked[:,:,:self.hsz] + unpacked[:,:,self.hsz:]
        dec_out = self.decoder(unpacked)
        return self.logsoftmax(dec_out)