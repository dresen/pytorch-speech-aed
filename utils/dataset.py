import torch
import torch.utils.data as tud
from audio import mfcc
from torch.nn.utils.rnn import pad_sequence


class AudioSequenceDataset(tud.Dataset):
    def __init__(self, list_of_paths, labels):
        super(AudioSequenceDataset, self).__init__()
        self.labels = labels
        self.files = list_of_paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        x = torch.FloatTensor(mfcc(fname))
        y = torch.IntTensor(self.labels[fname])

        return x.size()[0], y.size()[0], x, y

    
def collate_sequences(batch):
    batch = sorted(batch, key=lambda x: (x[0], x[1]))
    srclengths = torch.IntTensor([x[0] for x in batch])
    tgtlengths = torch.IntTensor([x[1] for x in batch])
    padded_feats = pad_sequence([x[2] for x in batch],
                                batch_first=True)
    padded_labels = pad_sequence([x[3] for x in batch],
                                batch_first=True)

    return srclengths, tgtlengths, padded_feats, padded_labels

    