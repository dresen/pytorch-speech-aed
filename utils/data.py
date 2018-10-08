from audio import audiosort
from text import make_char_int_maps, labels_from_string
from dataset import AudioSequenceDataset, collate_sequences
import sys


def format_data(audiolist, reflist, sym2int, name='train', partition={}, labels={}):
    """Formats and partitions the data based on aligned file lists. The function
    can be called multiple times with the a different name and the partition
    is just extended.
    
    Arguments:
        audiolist {list} -- List of paths to audio files
        reflist {list} -- List of reference transcriptions (text)
        sym2int {dict} -- Maps characters to integers
    
    Keyword Arguments:
        name {str} -- Name of the partition (default: {'train'})
        partition {dict} -- Maps samples to a partition (default: {{}})
        labels {dict} -- Maps labels to audio  (default: {{}})
    
    Returns:
        [dict,dict] -- the data partition table and an audio-to-reference table
    """
    # Make sure we've got the right amount of data
    num_audio = len(audiolist)
    num_refs = len(reflist)
    if num_audio != num_refs:
        common_len = min(num_audio, num_refs)
        audiolist = audiolist[:common_len]
        reflist = reflist[:common_len]
        print("Truncated the data set to {} utterances".format(common_len))
    # Don't overwrite
    if name in partition or name in labels:
        sys.exit("{} is already in the partition or labels dict. Aborting".format(name))
    
    partition[name] = audiolist
    skipped = 0
    for idx, x in zip(audiolist, reflist):
        if idx in labels:
            #print("Skipping {} ...".format(idx))
            skipped += 1
        else:
            labels[idx] = labels_from_string(x, sym2int)
    
    print("Skipped {} files".format(skipped))

    return partition, labels


if __name__ == "__main__":
    import torch
    from random import sample
    import torch.utils.data as tud
    from torch.nn.utils.rnn import pad_sequence
    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    translist = [open(x).read().strip() for x in reflist]
    print("Test audiosort function with references")
    # Sample randomly because an4 is already sorted
    idxs = sample(range(0,len(audiolist)), 10)
    testlist = [audiolist[x] for x in idxs]
    testref = [translist[x] for x in idxs]
    sortedlist = audiosort(testlist, list_of_references=testref, return_duration=True)
    print("Random -> sorted")
    for e in zip(testlist, sortedlist):
        print("{}\t{}".format(e[0],e[1]))

    sym2int, int2sym = make_char_int_maps(testref)
    partition, labels = format_data(testlist, testref, sym2int)

    partition, labels = format_data(testlist, testref, sym2int, 
                                    name='eval', partition=partition, labels=labels)
    print("Train:", partition['train'][:3])
    print("Eval:", partition["eval"][:3])
    print("Labels", list(labels.items())[:3])

    print("test dataset class")
    # Some training parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # tud.cudnn.benchmark = True    # Why?
    params = {'batch_size':2,
              'shuffle':False,
              'num_workers':2,
              'collate_fn':collate_sequences}


    trainset = AudioSequenceDataset(partition['train'], labels)
    
    traingenerator = tud.DataLoader(trainset, **params, )
    n = 0
    for xlens, ylens, xs, ys in traingenerator:
        print(xlens)
        print(ylens)
        print(xs.size())
        print(ys)
        break
