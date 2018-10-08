from audio import audiosort
from text import make_char_int_maps, labels_from_string
from dataset import AudioSequenceDataset as Dataset
import torch
import torch.utils.data as tud



def format_data(audiolist, reflist, name='train', partition={}, labels={}):
    """Formats and partitions the data based on aligned file lists. The function
    can be called multiple times with the a different name and the partition
    is just extended.
    
    Arguments:
        audiolist {list} -- list of paths to audio files
        reflist {list} -- list of reference transcriptions (text)
    
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
    sym2int, int2sym = make_char_int_maps(reflist)
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
    from random import sample
    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]

    print("Test audiosort function with references")
    # Sample randomly because an4 is already sorted
    idxs = sample(range(0,len(audiolist)), 10)
    testlist = [audiolist[x] for x in idxs]
    testref = [reflist[x] for x in idxs]
    sortedlist = audiosort(testlist, list_of_references=testref, return_duration=True)
    print("Random -> sorted")
    for e in zip(testlist, sortedlist):
        print("{}\t{}".format(e[0],e[1]))

    partition, labels = format_data(audiolist, reflist)

    partition, labels = format_data(audiolist, reflist, name='eval', partition=partition, labels=labels)
    print("Train:", partition['train'][:3])
    print("Eval:", partition["eval"][:3])
    print(list(labels.items())[:3])
    print(len(labels))

    print("test dataset class")

    # Some training parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # tud.cudnn.benchmark = True    # Why?
    params = {'batch_size':1,
              'shuffle':False,
              'num_workers':2}


    trainset = Dataset(partition['train'], labels)
    traingenerator = tud.DataLoader(trainset, **params)

    for iter in range(3):
        for batched_audio, batched_labels in traingenerator:
            print(batched_audio, batched_labels)