"""This script shows how to use the library to train speech recognisers
end-to-end with encoder-decoder models with attention
"""

import sys
import torch

import torch.utils.data as tud
from random import sample

import utils.data as data
import utils.audio as audio
from utils.voc import generate_char_voc, PAD_token
from utils.dataset import AudioDataset, Collate
from models.gru import BaseGru
from models.decoder import LuongAttentionDecoderRNN as AttentionDecoder
from train import train_attention 

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


clip = 50.0  # Unsure whether this is a good value
teacher_forcing_ratio = 1.0 # Not tuned
learning_rate = 0.001       # Not tuned
decoder_learning_ratio = 5.0 # 
model_name = 'cb_model'
attn_model = 'scaled_dot'
hidden_size = 400	        # Not tuned
encoder_nlayers = 2	        # Not tuned
decoder_nlayers = 2	        # Not tuned
dropout = 0.1	            # Not tuned
batch_size = 8


# Set random seeds
torch.manual_seed(123)

audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
translist = [open(x).read().strip() for x in reflist]

# Already sorted, but good practice imo
sortedlist = audio.audiosort(audiolist, list_of_references=translist)
trainlist, trainref = zip(*sortedlist)

# Define the vocabulary, integer maps and parition the data
voc = generate_char_voc(trainref, "TRAIN", mode='enc-dec')
partition, labels = data.format_data(trainlist, trainref, voc)

# Create the torch formatted training data and data functions
trainset = AudioDataset(partition['train'], labels, audio.mfcc_delta)
# Longest_first is required when we pack the input
batch_fn = Collate(PAD_token, longest_first=True )
bsz = 8
params = {'batch_size': bsz,
            'shuffle':False,
            'num_workers':2,
            'collate_fn':batch_fn,
            'drop_last': True}

embedding = torch.nn.Embedding(voc.num_words, hidden_size, padding_idx=PAD_token)

datagenerator = tud.DataLoader(trainset, **params)
encoder = BaseGru(120, hidden_size, encoder_nlayers, dropout=0.1, bidi=True)
decoder = AttentionDecoder(attn_model, embedding, hidden_size, hidden_size, 
                           voc.num_labels, decoder_nlayers, dropout=dropout)


encoder_optimiser = torch.optim.SGD(encoder.parameters(),
                            lr=learning_rate)
decoder_optimiser = torch.optim.SGD(decoder.parameters(),
                            lr=learning_rate * decoder_learning_ratio)

try:

    train_attention("traintest", "AN4", datagenerator, voc,
                    encoder, decoder, encoder_optimiser, decoder_optimiser, 
                    "an4test", epochs=20, batch_size=bsz, 
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    print_every=2, save_every=0, clip=clip, device=device)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    sys.exit()