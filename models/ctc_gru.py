"""GRU-based sequence models for CTC training
"""

import torch
import torch.nn.functional as F
from models.ctc_lstm import SequenceLinear

class CTCgru(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nlayers, dropout=0, bidi=False):
        super(CTCgru, self).__init__()
        self.isz = input_size      # Number of features
        self.osz = output_size + 1 # Remember to add "blank" to the output size if we use CTCLoss
        self.hsz = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.encoder = torch.nn.GRU(self.isz,self.hsz, dropout=self.dropout, 
        batch_first=True, bidirectional=bidi)

        self.decoder = SequenceLinear(self.hsz, self.osz)

        self.logsoftmax = torch.nn.LogSoftmax(2)


    def forward(self, sequences):
        enc_out, _ = self.encoder(sequences)
        dec_out = self.decoder(enc_out)
        return self.logsoftmax(dec_out)

        

class CTCgrup(CTCgru):
    def __init__(self, *args, **kwargs):
        super(CTCgrup, self).__init__(*args, **kwargs)
        self.pack_unpack = True


    def forward(self, sequences, sequence_lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths, batch_first=True)
        enc_out, _ = self.encoder(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        dec_out = self.decoder(unpacked)
        return self.logsoftmax(dec_out)