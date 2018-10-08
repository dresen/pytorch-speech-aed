import torch
import torch.utils.data as tud
from audio import mfcc

class AudioSequenceDataset(tud.Dataset):
    def __init__(self, list_of_ids, labels):
        super(AudioSequenceDataset, self).__init__()
        self.labels = labels
        self.ids = list_of_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        fname = self.ids[index]
        X = mfcc(fname)
        y = self.labels[fname]

        return X, y

    
        