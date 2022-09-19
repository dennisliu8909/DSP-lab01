'''
@Modified by Paul Cho; 10th, Nov, 2020

For NTHU DSP Lab 2022 Autumn
'''

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct

from Lab1_functions_student import pre_emphasis, STFT, mel2hz, hz2mel, get_filter_banks

filename = './audio.wav'
source_signal, sr = sf.read(filename) #sr:sampling rate
print('Sampling rate={} Hz.'.format(sr))

### hyper parameters
frame_length = 512                    # Frame length(samples)
frame_step = 256                      # Step length(samples)
emphasis_coeff = 0.95                 # pre-emphasis para
num_bands = 12                        # Filter number = band number
num_FFT = frame_length                # FFT freq-quantization
freq_min = 0
freq_max = int(0.5 * sr)
signal_length = len(source_signal)    # Signal length

# number of frames it takes to cover the entirety of the signal
num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step))

##########################
'''
Part I:
(1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
(2) Pre-emphasize the source signal with pre_emphasis()
(3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
(4) Plot the two spectrograms together to observe the effect of pre-emphasis

hint for plotting:
you can use "plt.subplots()" to plot multiple figures in one.
you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
'''
#YOUR CODE STARTS HERE:
spectrum1 = STFT(source_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, False)
signal_pre_emphasis = pre_emphasis(source_signal, 0.95)
spectrum2 = STFT(signal_pre_emphasis, num_frames, num_FFT, frame_step, frame_length, signal_length, False)

# plt.plot(spectrum1)
# plt.show()

# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.pcolor(spectrum1)
# plt.suptitle("plot exercise")
# ax0.set_xlabel("freq")
# ax0.set_ylabel("STFT spectrum")
# ax1.pcolor(spectrum2)
# ax1.set_xlabel("freq")
# ax1.set_ylabel("STFT spectrum")
# plt.show()

#YOUR CODE ENDS HERE;
##########################

'''
Head to the import source 'Lab1_functions_student.py' to complete these functions:
mel2hz(), hz2mel(), get_filter_banks()
'''
# get Mel-scaled filter
fbanks = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max)
xaxis = np.arange(0, num_FFT / 2 + 1, 1)
for i in range (0, num_bands):
    plt.plot(xaxis, fbanks[i])
plt.xlabel("freq")
plt.ylabel("mel-scale filter banks")
plt.show()
##########################
'''
Part II:
(1) Convolve the pre-emphasized signal with the filter
(2) Convert magnitude to logarithmic scale
(3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
    (already implemented for you, just notice this step is here and skip to the next step)
(4) Plot the filter banks alongside the MFCC
'''
#YOUR CODE STARTS HERE:
signal_after_filter = np.dot(fbanks, spectrum2)
for i in range (0, num_bands):
    signal_after_filter[i] = np.log(signal_after_filter[i])
# plt.pcolor(signal_after_filter)
# plt.show()
# print(signal_after_filter.shape)
features = signal_after_filter

# step(3): Discrete Cosine Transform
MFCC = dct(features.T, norm = 'ortho')[:,:num_bands]
# equivalent to Matlab dct(x)
# The numpy array [:,:] stands for everything from the beginning to end.

plt.pcolor(MFCC.T)
plt.show()


#YOUR CODE ENDS HERE;
##########################
