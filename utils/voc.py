# Default word tokens
PAD_token = 0  # Used for padding short sequences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    """A class that can contain the vocabulary we want to model and build the 
    integer maps we will need to encode and decode labels (ints)
    
    Returns:
        Voc -- The vocabulary object
    """
    def __init__(self, name, mode='ctc', space_map=('-')):
        self.name = name
        self.trimmed = False
        self._mode = mode
        self._space_map = list(space_map)
        self._init_maps()

    def _init_maps(self):
        """Initialise the integer mappings with special symbols that must be present and
        map all symbols in self._space_map to whatever ' ' is mapped to (default is '1')
        """
        # Maps
        self.index2word = {}
        self.word2index = {}
        self.word2count = {}
        # Counts
        self.num_words = 0
        self.num_labels = 0
        
        # Specific inits
        if self._mode == 'ctc':
            # Initialise mappings with space (Reserve zero for the CTC blank symbol)
            self.index2word[1] = ' '
            self.word2index[' '] = 1 
            self.word2count[' '] = 0
            self.num_words = 1
            self.num_labels = 1
            # Initialise sym2int mapping with special handling of 'space_map'
            # (don't increase the number of labels because they all map to the same label)
            for word in self._space_map:
                if word != ' ':
                    self.word2index[word] = self.num_labels
                    self.num_words += 1
                    self.word2count[word] = 0 # None of them are 'seen' yet
        elif self._mode == 'enc-dec':
            # Ignore space_map in this case
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD
            self.num_labels = 3 # They will be the same as num_words in this case
        else:
            raise ValueError("Unknown vocabulary mode - use 'ctc' or 'enc-dec'")

    def add_word(self, word):
        """Add the input word to the vocabulary and/or count the occurence. 
        The "word" can also be a character or other symbol
        
        Arguments:
            word {str} -- a string of characters
        """
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
            self.num_labels += 1
        else:
            self.word2count[word] += 1

    def add_sequence(self, sequence):
        """Add the elems in the presegmented input sequence
        
        Arguments:
            sequence {list} -- A presegmented input sequence 
        """
        for elem in sequence:
            self.add_word(elem)

    def add_sentence(self, word):
        """Segments the input by space and adds the 'words' 
        
        Arguments:
            word {str} -- A string of words
        """
        self.add_sequence(sequence.split(' '))

    def labels_from_sequence(self, sequence):
        return [self.word2index.get(x) for x in sequence]

    def labels_from_sentence(self, sentence):
        return self.labels_from_sequence(sentence.split(' '))
    
    def labels_from_chars(self, char_string):
        return self.labels_from_sequence(list(char_string))

    def sequence_from_labels(self, label_sequence):
        return [self.index2word.get(x) for x in label_sequence]
        
    def sentence_from_labels(self, label_sequence):
        return ' '.join(self.labels_from_sequence(label_sequence))

    def string_from_labels(self, label_sequence):
        return ''.join(self.sequence_from_labels(label_sequence))

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self._init_maps()

        for word in keep_words:
            self.add_word(word)

    def __len__(self):
        return len(self.word2index)


def generate_char_voc(text_corpus, name, mode='ctc', space_map=['-']):
    """Utility to build the vocabulary
    
    Arguments:
        text_corpus {list} -- List or other sequence of strings
        name {str} -- The name of the corpus or other descriptor
    
    Keyword Arguments:
        space_map {list} -- Input symbols we want mapped to the space token (default: {['-']})
    
    Returns:
        Voc -- The vocabulary and integer maps
    """
    voc = Voc(name, mode, space_map)
    for text in text_corpus:
        voc.add_sequence(list(text))
    return voc


if __name__ == '__main__':
    reflist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.reference").readlines()]
    print("Run test on {}".format(reflist[0]))
    ref = open(reflist[0], 'r').read()

    voc = generate_char_voc(ref, "TEST CTC")
    print(voc.word2index)
    print(voc.index2word)
    print(voc.name, len(voc))
    print("ORG:     ", ref)
    print("ORG->LBL:", voc.labels_from_chars(ref))
    print("LBL->STR:", voc.string_from_labels(voc.labels_from_chars(ref)))

    voc = generate_char_voc(ref, "TEST ENC-DEC", mode='enc-dec')
    print(voc.word2index)
    print(voc.index2word)
    print(voc.name, len(voc))
    print("ORG:     ", ref)
    print("ORG->LBL:", voc.labels_from_chars(ref))
    print("LBL->STR:", voc.string_from_labels(voc.labels_from_chars(ref)))

    # Should fail
    voc = generate_char_voc(ref, "TEST ENC-DEC", mode='Not-a-mode')
