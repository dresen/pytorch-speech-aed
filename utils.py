"""
This module contains all the utility functions needed to run LSTM-CTC ASR
with pytorch
"""

import librosa as lr


def make_char_int_maps(textcorpus, zero_map=(' ', '-')):
    """Create the char map needed for CTC. zero_map is 'space'"""
    char2int = {'<space>': 0}
    int2char = {0: '<space>'}
    i = 0
    for text in textcorpus:
        for sym in list(text):
            if sym not in char2int and sym not in zero_map:
                i += 1
                char2int[sym] = i
                int2char[i] = sym
    return (char2int, int2char)


def labels_from_string(string, sym2int, zero_map=(' ', '-')):
    """Apply a map to a string to get a transformed (integer) sequence"""
    return [0 if x in zero_map else sym2int.get(x) for x in list(string)]


def mfcc(audiofile, samplerate=8000, num_cepstra=40, start=0, stop=None):
    """Extracts 40-dim MFCC vectors from a sound file - options for specifying
    a part of the sound file"""

    # Load ignores duration if it is None
    duration = None if stop is None else stop - start
    signal, _ = lr.load(audiofile, sr=samplerate, offset=start, duration=duration,)
    signal = lr.feature.melspectrogram(signal, sr=samplerate, n_mels=128)
    log_signal = lr.logamplitude(signal)
    feats = lr.feature.mfcc(S=log_signal, n_mfcc=num_cepstra)
    return feats.T  # Transpose so we iterate over samples in loops



if __name__ == "__main__":
    print("testing MFCC extraction on an121-fjdn-b.wav from an4")
    testing = mfcc("an4_utils_test.wav")
    print (testing)
    