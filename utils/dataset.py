import torch
import torch.utils.data as tud

from torch.nn.utils.rnn import pad_sequence


class AudioDataset(tud.Dataset):
    """A dataset to parallelise batching to train DNNs
    
    Arguments:
        tud {Dataset} -- Base Class
    
    Returns:
        AudioDataset object -- Defines how to load data and serve data in batches
    """ 
    def __init__(self, list_of_paths, labels, wavtransform):
        """Initialisation
        
        Arguments:
            list_of_paths {list} -- List of paths to audio files or IDs
            labels {dict} -- Mapping from path/filename/ID to target label sequence
        """

        super(AudioDataset, self).__init__()
        self.labels = labels
        self.files = list_of_paths
        self.wavtransform = wavtransform

    def __len__(self):
        """Returns the number of sequences in the data set
        
        Returns:
            int -- Number of sequences in the data set
        """
        return len(self.files)

    def __getitem__(self, index):
        """How to get a single sequence from the data set
        
        Arguments:
            index {int} -- Index to the list of paths/IDs
        
        Returns:
            tuple(IntTensor, IntTensor, FloatTensor, FloatTensor) -- [description]
        """
        fname = self.files[index]
        # shape is num samples * feature dim
        x = torch.FloatTensor(self.wavtransform(fname))
        # shape is num samples * 1
        y = torch.IntTensor(self.labels[fname])

        return x.size()[0], y.size()[0], x, y

    
    
class Collate:
    """Class that wraps a function that batches training data
    
    Returns:
        Collate object -- Can batch data in different modes
    """
    def __init__(self, padding_value, longest_first=False):
        self.padding = padding_value
        self.reverse = longest_first

    def collate_sequences(self, batch):
        batch = sorted(batch, key=lambda x: (x[0], x[1]), reverse=self.reverse)
        srclengths = torch.IntTensor([x[0] for x in batch])
        tgtlengths = torch.IntTensor([x[1] for x in batch])
        padded_feats = pad_sequence([x[2] for x in batch],
                                    batch_first=True)
        padded_labels = pad_sequence([x[3] for x in batch],
                                     batch_first=True, padding_value=self.padding)

        return srclengths, tgtlengths, padded_feats, padded_labels
        
    def __call__(self, batch):
        return self.collate_sequences(batch)


if __name__ == "__main__":
    import torch
    import torch.utils.data as tud
    # from torchnlp.samplers import BPTTBatchSampler
    from random import sample
    from audio import audiosort, mfcc
    from voc import generate_char_voc
    from data import format_data #, make_char_int_maps
    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    translist = [open(x).read().strip() for x in reflist]
    # Sample randomly because an4 is already sorted
    idxs = sample(range(0,len(audiolist)), 10)
    testlist = [audiolist[x] for x in idxs]
    testref = [translist[x] for x in idxs]
    sortedlist = audiosort(testlist, list_of_references=testref)
    testlist, testref = zip(*sortedlist)
    voc = generate_char_voc(testref, "LE TEST")
    #sym2int, int2sym = make_char_int_maps(testref, offset=1)
    partition, labels = format_data(testlist, testref, voc)

    print("test dataset class")
    trainset = AudioDataset(partition['train'], labels, wavtransform=mfcc)
    ctc_batch_fn = Collate(-1)
    params = {'batch_size': 2,
              'shuffle':False,
              'num_workers':2,
              'collate_fn':ctc_batch_fn}


    traingenerator = tud.DataLoader(trainset, **params)
    n = 0
    for xlens, ylens, xs, ys in traingenerator:
        print(xlens.size())
        print(ylens.size())
        print(xs.size())
        print(ys.size())
        break