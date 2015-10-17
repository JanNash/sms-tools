﻿from scipy.fftpack import fft
import numpy as np

"""
A3-Part-2: Optimal zero-padding

Given a sinusoid, write a function that computes the DFT of the sinusoid after zero-padding and returns
the positive half of the magnitude spectrum (in dB). Zero-padding needs to be done such that one of 
the bin frequencies of the DFT coincides with the frequency of the sinusoid. Choose the minimum 
zero-padding length for which this condition is satisfied. 

The input arguments are the sinusoid x of length M, sampling frequency fs and the frequency of the 
sinusoid f. The output is the positive half of the magnitude spectrum mX computed using the N point 
DFT (N >= M) of x after zero-padding x to length N appropriately as required. 

To get the positive half of the spectrum, first compute the N point DFT of the zero-padded input signal 
(for this you can use the fft function of scipy.fftpack, which is already imported in this script). 
Consider only the first (N/2)+1 samples of the DFT and compute the magnitude spectrum of the positive 
half (in dB) as mX = 20*log10(abs(X[:(N/2)+1])), where X is the N point DFT of the zero-padded input.

For this exercise, you can assume that the frequency of the sinusoid f is a positive integer and a 
factor of the sampling rate fs. The input parameters will be given in such a way that N will be even.
Note that the numerical value of f is an integer but the data type is float, for example 1.0, 2.0, 
55.0 etc. This is to avoid issues in python related with division by a integer.

Due to the precision of the FFT computation, the zero values of the DFT are not zero but very small
values < 1e-12 (or -240 dB) in magnitude. For practical purposes, all values with absolute value less 
than 1e-6 (or -120 dB) can be considered to be zero. 

HINT: One of the DFT bin frequencies coincides with the frequency f of a sinusoid when the DFT size 
(N in this question) contains exactly an integer number of periods of the sinusoid. For example, 
if f = 100 Hz and fs = 1000 Hz, one period of the sinusoid has 10 samples. Then given a signal of length 
M = 25 samples, there are 2.5 periods in it. The minimum zero-padding length here would be 5 samples 
(0.5 period), so that the DFT size N = 30 corresponds to 3 periods of a sinusoid of frequency f=100 Hz.

Test case 1: For a sinusoid x with f = 100 Hz, M = 25 samples and fs = 1000 Hz, you will need to 
zero-pad by 5 samples and compute an N = 30 point DFT. In the magnitude spectrum, you can see a 
maximum value at bin index 3 corresponding to the frequency of 100 Hz. The output mX you return is 
16 samples in length. 

Test case 2: For a sinusoid x with f = 250 Hz, M = 210 samples and fs = 10000 Hz, you will need to 
zero-pad by 30 samples and compute an N = 240 point DFT. In the magnitude spectrum, you can see a 
maximum value at bin index 6 corresponding to the frequency of 250 Hz. The output mX you return is 
121 samples in length. 

"""
def optimalZeropad(x, fs, f):
    """
    Inputs:
        x (numpy array) = input signal of length M
        fs (float) = sampling frequency in Hz
        f (float) = frequency of the sinusoid in Hz
    Output:
        The function should return
        mX (numpy array) = The positive half of the DFT spectrum of the N point DFT after zero-padding 
                        x appropriately (zero-padding length to be computed). mX is (N/2)+1 samples long
    """
    ## Your code here
    periodLength = fs // f
    signalLength = len(x)

    periods = float(signalLength) / float(periodLength)

    n = int(signalLength + ((np.ceil(periods) - periods) * periodLength))

    dft = fft(x, n)

    return 20 * np.array(np.log10(dft[:((n / 2) + 1)]))


# def test_minimizeEnergySpreadDFT(f, fs, m):
#     from workspace.A2.A2Part1 import genSine
#     from random import uniform
#
#     t = uniform(0, 5)
#     amp = uniform(0, 10)
#     phi = uniform(-10, 10)
#
#     print('Generating sinusoid at sampling rate {fs} Hz with a duration of {t} seconds...'
#           ''.format(fs=fs, t=t))
#     print('Sinusoid has an amplitude of {amp}, a frequency of {f} Hz, and a phase offset of {phi}...'
#           ''.format(amp=amp, f=f, phi=phi))
#     x = genSine(A=amp, f=f, phi=phi, fs=fs, t=t)
#
#     print('Only using the first {m} samples of the sinusoid...'.format(m=m))
#     mx = optimalZeropad(x=x[:m], fs=fs, f=f)
#
#     length = len(mx)
#     print('Length of mx: {length}'.format(length=length))
#
#     maximum = max(mx)
#     ind = mx.argmax(axis=0)
#     print('Maximum of mx: {maximum} dB at index {index}'.format(maximum=maximum, index=ind))
#
#
# test_minimizeEnergySpreadDFT(f=100, fs=1000, m=25)
# test_minimizeEnergySpreadDFT(f=250, fs=10000, m=210)