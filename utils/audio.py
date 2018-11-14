"""
This module contains all the utility functions needed to run LSTM-CTC ASR
with pytorch
"""

import librosa as lr
import warnings

import numpy as np

def raw_mfcc(audiofile, samplerate=8000, num_cepstra=40, start=0.0, stop=0.0, delta=False):
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
    return lr.feature.mfcc(S=log_signal, n_mfcc=num_cepstra)

def mfcc(audiofile, samplerate=8000, num_cepstra=40, start=0.0, stop=0.0):
    return raw_mfcc(audiofile, samplerate, num_cepstra, start, stop).T


def mfcc_delta(audiofile, samplerate=8000, num_cepstra=40, start=0.0, stop=0.0):
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
    # shape is (num_cepstra, n_samples)
    feats = raw_mfcc(audiofile, samplerate, num_cepstra, start, stop)
    with warnings.catch_warnings():
        # Catch FutureWarning in scipy that we can't fix
        warnings.simplefilter("ignore")
        d = lr.feature.delta(feats)
        dd = lr.feature.delta(feats, order=2)
        feats = np.vstack((feats, d, dd))
    # shape is (num_samples, num_cepstra*3)
    return feats.T  # Transpose so we iterate over samples in loops


def superframes(mat, left_context=2, right_context=0):
    """Stack features vectors to superframes. Matrix dimensions
    must be (seqlen, feature_len)
    
    Arguments:
        mat {np.array} -- A (seqlen,feature_len) matrix
    
    Keyword Arguments:
        left_context {int} -- Left window size (default: {2})
        right_context {int} -- Right window size (default: {0})
    
    Returns:
        np.array -- A (seqlen,feature_len*(left_context+right_context)) matrix
    """

    first, last = mat[:1], mat[-1:]
    stack = [mat]
    # We use two loops because the left and right context
    # can be different sizes
    for i in range(1, left_context + 1):
        roll = np.roll(mat, i, axis=0)
        # Broadcast the first vector to the beginning of the new matrix
        # Otherwise the last i-num vectors appear in the beginning
        roll[:i] = first
        stack.append(roll)
    # Add -1 as step size to range so we iterate over the sequence in reverse
    for i in range(-1, -1 * (right_context + 1), -1):
        roll = np.roll(mat, i, axis=0)
        roll[i:] = last
        stack.append(roll)
    return np.hstack(stack)


def cms(matrix):
    """Cepstral Mean subtraction
    
    Arguments:
        matrix {np.array} -- A feature matrix with (seqlen,feature_len) dimensions
    
    Returns:
        np.array -- A matrix of the same dimension where the mean has been subtracted
    """

    return matrix - matrix.mean(axis=0, keepdims=True)


def cmvn(matrix):
    """Cepstral Mean and Variance normalisation. Not frequently used
    
    Arguments:
        matrix {np.array} -- A feature matrix with (seqlen,feature_len) dimensions
    
    Returns:
        np.array -- A matrix of the same dimensions with zero mean and unit variance
    """
    return cms(matrix) / matrix.std(axis=0)


def decimate(matrix, skip=2):
    """Subsample elements from a 2D sequence, e.g. go from 10ms sample rate
    to a 30ms sample rate if skip=2.
    
    Arguments:
        matrix {np.array} -- A feature matrix with (seqlen,feature_len) dimensions
    
    Keyword Arguments:
        skip {int} -- Number of consecutive feature vectors to skip (default: {2})
    
    Returns:
        np.array -- A (seqlen/skip+1,feature_len) feature matrix
    """

    return matrix[skip::skip+1]


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
    _ = cmvn(testing) # also tests cms
    print(decimate(testing))
    print(testing[:3], "\nShape is {0} x {1} (samples x features)".format(testing.shape[0],testing.shape[1]))
    stack_testing = superframes(np.reshape(np.arange(40), (10,4)), left_context=2, right_context=0)
    print(stack_testing, stack_testing.shape)
    stack_testing_dec = decimate(stack_testing)
    print(stack_testing_dec, stack_testing_dec.shape)
    print("Testing sort function")
    # Sample randomly because an4 is already sorted
    testlist = sample(audiolist, 10)
    sortedlist = audiosort(testlist, return_duration=True)
    print("Random -> sorted")
    for e in zip(testlist[:3], sortedlist[:3]):
        print("{}\t{}".format(e[0],e[1]))
    testing2 = mfcc_delta(audiolist[1])
    print("Shape is {0} (samples x features)".format(testing2.shape))

    
