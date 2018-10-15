import sys


def format_data(audiolist, reflist, sym2int, texttransform, name='train', partition={}, labels={}):
    """Formats and partitions the data based on aligned file lists. The function
    can be called multiple times with the a different name and the partition
    is just extended.
    
    Arguments:
        audiolist {list} -- List of paths to audio files
        reflist {list} -- List of reference transcriptions (text)
        sym2int {dict} -- Maps characters to integers
        texttransfomr {function} -- function that maps a text string to a label string
    
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
            labels[idx] = texttransform(x, sym2int)
    
    print("Skipped {} files".format(skipped))

    return partition, labels


if __name__ == "__main__":
    from random import sample
    import torch
    import torch.utils.data as tud
    from torch.nn.utils.rnn import pad_sequence
    from text import labels_from_string, make_char_int_maps
    from audio import audiosort
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
    testlens, testlist, testref = zip(*sortedlist)
    sym2int, int2sym = make_char_int_maps(testref, offset=1)
    print(sym2int)
    partition, labels = format_data(testlist, testref, sym2int, labels_from_string)

    partition, labels = format_data(testlist, testref, sym2int, labels_from_string, 
                                    name='eval', partition=partition, labels=labels)
    print("Train:", partition['train'][:3])
    print("Eval:", partition["eval"][:3])
    print("Labels", list(labels.items())[:3])

