import re
from voc import Voc

def make_char_int_maps(textcorpus, zero_map=(' ', '-')):
    """Create the mappings from characters to integers and vice versa. Note that there's a 
    Problem with hyphens and they must always be zero-mapped
    
    Arguments:
        textcorpus {List} -- A list of transcriptions
    
    Keyword Arguments:
        zero_map {tuple} -- Character symbols we want to map to 0 (default: {(' ', '-')})
    
    Returns:
        tuple(dict, dict) -- character to int and int to character mappings
    """
    char2int = {'<space>': 0}
    int2char = {0: ' '}
    i = 0
    for text in textcorpus:
        # split reference to list of symbols
        for sym in list(text):
            if sym not in char2int and sym not in zero_map:
                i += 1
                char2int[sym] = i
                int2char[i] = sym

    return (char2int, int2char)


def labels_from_string(string, sym2int, zero_map=(' ', '-')):
    """Apply a map to generate integers. Works on single characters
    and is reversible (you can use it to map a list of integers to 
    a list of characters that you can .join() to a string sequence)
    
    Arguments:
        string {str} -- A case-normalised transcription 
        sym2int {dict} -- Maps from char to int
    
    Keyword Arguments:
        zero_map {tuple} -- Characters that we want to ignore and map to 0 (default: {(' ', '-')})
    
    Returns:
        list -- List of integers (or chars)
    """
    return [0 if x in zero_map else sym2int.get(x) for x in list(string)]

def normalise_string(s, zero_map=(' ', '-')):
    s = s.lower().strip()
    s = re.sub(r"[-.!?]", r" ", s)
    s = re.sub(r" +", r" ", s)
    return s
    


if __name__ == '__main__':
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    print("Run test on {}".format(reflist[0]))
    ref = open(reflist[0], 'r').read()
    sym2int, int2sym = make_char_int_maps(ref)
    print(sym2int)
    print(ref)
    labels = labels_from_string(ref, sym2int)
    print(labels)
    print(''.join(labels_from_string(labels, int2sym)))


    