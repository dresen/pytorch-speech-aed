"""This script shows how to use the library to train speech recognisers
end-to-end
"""

import sys
import torch
import torch.utils.data as tud
from random import sample
import utils.data as data
import utils.audio as audio
from utils.dataset import AudioDataset, Collate
from utils.voc import generate_char_voc
from models.ctc import CTClstmp
from train import train_ctc

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
translist = [open(x).read().strip() for x in reflist]

# Already sorted, but good practice imo
sortedlist = audio.audiosort(audiolist, list_of_references=translist)
trainlist, trainref = zip(*sortedlist)

voc = generate_char_voc(trainref, "TRAIN", mode='ctc')
partition, labels = data.format_data(trainlist, trainref, voc)

trainset = AudioDataset(partition['train'], labels, audio.mfcc)
# Longest_first is required when we pack the input
ctc_batch_fn = Collate(-1, longest_first=True )
bsz = 8
params = {'batch_size': bsz,
            'shuffle':False,
            'num_workers':2,
            'collate_fn':ctc_batch_fn,
            'drop_last': True}

datagenerator = tud.DataLoader(trainset, **params, )
model = CTClstmp(40, len(voc), 120, 2)
model.to(device)
optimiser = torch.optim.SGD(model.parameters(),
                            lr=0.01)
try:

    train_ctc("traintest", "AN4", datagenerator, voc,
              model,optimiser, "an4test", epochs=20, batch_size=bsz,
              print_every=2, save_every=0, clip=5.0, device=device)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.exit()