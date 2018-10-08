"""
This module contains all the utility functions needed to run LSTM-CTC ASR
with pytorch
"""

import librosa as lr


def mfcc(audiofile, samplerate=8000, num_cepstra=40, start=0.0, stop=0.0):
    """Extracts MFCC features from an audio file with optional segmentation
    
    Arguments:
        audiofile {string} -- Path to audio file
    
    Keyword Arguments:
        samplerate {int} -- Sample rate we want to downsample to (default: {8000})
        num_cepstra {int} -- Feature length (default: {40})
        start {float} -- Optional start time (default: {0.0})
        stop {float} -- Optional stop time (default: {0.0})
    
    Returns:
        np.ndarray -- Samples X Features matrix of MFCC feature
    """

    # Load ignores duration if it is None
    duration = None if stop == 0.0 else stop - start
    signal, _ = lr.load(audiofile, sr=samplerate, offset=start, duration=duration,)
    signal = lr.feature.melspectrogram(signal, sr=samplerate, n_mels=128)
    # Takes the log_10
    log_signal = lr.power_to_db(signal)
    # Matrix shape is (num_cepstra, n_samples)
    feats = lr.feature.mfcc(S=log_signal, n_mfcc=num_cepstra)
    return feats.T  # Transpose so we iterate over samples in loops


def audiolen(wavfile):
    """Get the length of the audio file. We use it to sort the data
    
    Arguments:
        wavfile {string} -- Path to the audio file   

    Returns:
        float -- Duration in seconds
    """
    return lr.get_duration(filename=wavfile)


def audiosort(list_of_files, list_of_references=None, return_duration=False):
    """Sorts a list of audio files based on their duration in seconds
    
    Arguments:
        list_of_files {List} -- A list with paths to audio files

    Keyword Arguments:
        list_of_references {List} -- A list of reference transcripts that are aligned with the audio
        return_duration {Bool} -- Whether to return duration for debugging (default: {False})
    Returns:
        List -- The sorted list with optional info
    """
    # Get the sorting info
    lengths = map(audiolen, list_of_files)
    
    if (list_of_references is not None):
        samplelist = zip(lengths, list_of_files, list_of_references)
    elif (list_of_references is None):
        samplelist = zip(lengths, list_of_files)
    # Actual sorting
    samplelist = sorted(samplelist, key=lambda x: x[0])

    return samplelist if return_duration else [x[1:] for x in samplelist]

    


if __name__ == "__main__":
    import os
    from random import sample
    audiolist = [x.strip() for x in open("/Users/akirkedal/workdir/speech/data/an4train.list").readlines()]
    print("Testing MFCC extraction on {}".format(audiolist[0]))
    testing = mfcc(audiolist[0])
    print(testing[:3], "\nShape is {0} x {1} (samples x features)".format(testing.shape[0],testing.shape[1]))
    print("Testing sort function")
    # Sample randomly because an4 is already sorted
    testlist = sample(audiolist, 10)
    sortedlist = audiosort(testlist, return_duration=True)
    print("Random -> sorted")
    for e in zip(testlist, sortedlist):
        print("{}\t{}".format(e[0],e[1]))
    
