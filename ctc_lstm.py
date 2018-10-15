import torch
import torch.nn.functional as F


class SequenceLinear(torch.nn.Linear):
    def __init__(self, input_size, output_size):
        super(SequenceLinear, self).__init__(input_size, output_size)

    def forward(self, x):

        # Assume the input is bsz, seqlen, featlen
        # or seqlen, bsz, featlen

        dim1, dim2 = x.size()[0], x.size()[1]
        x = x.reshape(dim1 * dim2, -1)
        y = F.linear(x, self.weight, self.bias)
        return y.reshape(dim1, dim2, -1)

class CTClstm(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, nlayers, dropout=0, bidi=False):
        super(CTClstm, self).__init__()
        self.isz = input_size      # Number of features
        self.osz = output_size + 1 # Remember to add "blank" to the output size if we use CTCLoss
        self.hsz = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.encoder = torch.nn.LSTM(self.isz,self.hsz,self.osz, dropout=self.dropout, 
        batch_first=True, bidirectional=bidi)

        self.decoder = SequenceLinear(self.hsz, self.osz)

        self.logsoftmax = torch.nn.LogSoftmax(2)


    def forward(self, sequences):
        enc_out, _ = self.encoder(sequences)
        dec_out = self.decoder(enc_out)
        return self.logsoftmax(dec_out)

        

class CTClstmp(CTClstm):
    def __init__(self, *args, **kwargs):
        super(CTClstmp, self).__init__(*args, **kwargs)


    def forward(self, sequences, sequence_lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, sequence_lengths, batch_first=True)
        enc_out, _ = self.encoder(packed)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        dec_out = self.decoder(unpacked)
        return self.logsoftmax(dec_out)
    
    
if __name__ == "__main__":
    import torch
    import torch.utils.data as tud
    from random import sample
    import utils.data as data
    import utils.text as text
    import utils.audio as audio
    from utils.dataset import AudioDataset, Collate
    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    translist = [open(x).read().strip() for x in reflist]
    # Sample randomly because an4 is already sorted
    idxs = sample(range(0,len(audiolist)), 10)
    testlist = [audiolist[x] for x in idxs]
    testref = [translist[x] for x in idxs]
    sortedlist = audio.audiosort(testlist, list_of_references=testref)
    testlist, testref = zip(*sortedlist)
 
    sym2int, int2sym = text.make_char_int_maps(testref, offset=1)
    partition, labels = data.format_data(testlist, testref, sym2int, text.labels_from_string)

    trainset = AudioDataset(partition['train'], labels, audio.mfcc)
    ctc_batch_fn = Collate(-1)
    bsz = 2
    params = {'batch_size': bsz,
              'shuffle':False,
              'num_workers':2,
              'collate_fn':ctc_batch_fn}

    print("test the models on a batch of data")
    ctc_loss = torch.nn.CTCLoss()
    allloss = 0
    traingenerator = tud.DataLoader(trainset, **params, )
    n = 0
    for xlens, ylens, xs, ys in traingenerator:
        print(xlens)
        print(ylens)
        print(xs.size())
        print(ys)
        print("CTClstm")
        model = CTClstm(40, len(sym2int), 50, 2)
        preds = model(xs)
        print(preds.size(), "(len(sym2int) was", len(sym2int))
        preds = preds.reshape(-1, bsz, len(sym2int)+1)
        print(preds.size(), "(len(sym2int) was", len(sym2int))
        loss = ctc_loss(preds, ys, xlens, ylens)
        loss.backward()
        break
    ctc_batch_fn = Collate(-1, longest_first=True)
    params = {'batch_size': bsz,
              'shuffle':False,
              'num_workers':2,
              'collate_fn':ctc_batch_fn}

    ctc_loss = torch.nn.CTCLoss()
    allloss = 0
    traingenerator = tud.DataLoader(trainset, **params, )
    for xlens, ylens, xs, ys in traingenerator:
        print("CTClstmp")
        model = CTClstmp(40, len(sym2int), 50, 2)
        preds = model(xs, xlens)
        print(preds.size(), "(len(sym2int) was", len(sym2int))
        preds = preds.reshape(-1, bsz, len(sym2int)+1)
        print(preds.size(), "(len(sym2int) was", len(sym2int))
        loss = ctc_loss(preds, ys, xlens, ylens)
        loss.backward()
        break