# RF and Optical Communication Course Project
# Abhishek Nair, Mohit Shrivastava, Mohammed Khandwawala

import numpy as np
from scipy.signal import spectrogram, hanning,convolve , chirp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from mpl_toolkits.mplot3d import Axes3D
import sys
from sklearn.cluster import k_means 
from scipy.stats import skew, kurtosis
import pandas as pd

# Frequencies at which external noise is added
f1 = 275
f2 = 1333
f3 = 4150
f4 = 9644
f5 = 21212
f6 = 40000
f7 = 20000

K=int(1e5)

# The signal is assumed to be gaussian in nature due to CLT
sample = np.random.normal(0,1,10*K)  # 1ms interval 

# Adding a chirp signal as a noise source
#t = np.linspace(0,1,1001)
#C = chirp(t,5000,1000,1)
#plt.plot(t,C)
#plt.show()

# Time signals 
t1 = np.linspace(0,10,10*K)
t2 = np.linspace(0,10,10*K)
t3 = np.linspace(0,10,10*K)
t4 = np.linspace(0,10,10*K)
t5 = np.linspace(0,10,10*K)
t6 = np.linspace(0,10,10*K+1)

# chirp signal
C = (0.5)*chirp(t6,f7,10,f6)

# sinusoidal signals
sino1 = (0.5)*np.sin(2*np.pi*f1*t1)
sino2 = (0.5)*np.sin(2*np.pi*f2*t2)
sino3 = (0.5)*np.sin(2*np.pi*f3*t3)
sino4 = (0.5)*np.sin(2*np.pi*f4*t4)
sino5 = (0.5)*np.sin(2*np.pi*f5*t5)


# Simulating gaussian noise, same characteristics as the signal
#x = np.random.uniform(0,10000 - 1000)

# Adding the sinusoidal and chirp signals at different positions 
sample[5*K:] = sample[5*K:]+sino1[:5*K]
sample[:5*K] = sample[:5*K]+sino2[:5*K]
sample[K:6*K] = sample[1*K:6*K]+sino3[:5*K]
sample[4*K:9*K] = sample[4*K:9*K]+sino4[:5*K]
sample[2*K:7*K] = sample[2*K:7*K]+sino5[:5*K]
sample[:] = sample[:]+C[:10*K]

# computing 512 point spectrogram with an overlap of 75%
# keeping overlap allows us to retain phase information in the spectrogram 

NFFT = 512
overlap = 0.25
overlap_samples = int(round(NFFT*overlap)) # overlap in samples

f, t , S = spectrogram(sample,nperseg=NFFT,noverlap=overlap_samples,nfft=NFFT)
print(S.shape)
S = S[:,:256]
t = range(256)
f = range(257)
# Compute average spectrum
avg_S = np.mean(S,axis=1)

# plot the spectrogram of the signal
plt.pcolormesh(t, f, abs(S), cmap=plt.get_cmap('jet'))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title(" Spectrogram of the input signal " )
plt.colorbar()
plt.show()
'''
# Applying the hanning window to the spectrogram in order to remove noise but retain interference
h = hanning(5)
han2d = np.sqrt(np.outer(h,h))
S = convolve(S,han2d,mode="same")

# Compute average spectrum, that is average intensity at every frequency
avg_S = np.mean(S,axis=1)

plt.pcolormesh(t, f, abs(S), cmap=plt.get_cmap('jet'))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title( " Spectrogram after application of hanning window " )
plt.colorbar()
plt.show()
'''
# implementing a thresholding operation
mean = np.mean(S, axis=1)
std = np.std(S)

print( mean.shape )
thresh = 3*std 

plt.plot( mean )
plt.plot( thresh*np.ones(len(mean)) )
plt.ylabel("Mean Intensity Values ")
plt.xlabel("Normalized Frequencies (0 - 256)")
plt.title("Mean Intensity vs Frequency")
plt.legend(["signal","decision threshold"])
plt.show()
'''
G = np.zeros(S.shape)
G[abs(abs(S) -  mean) >=  thresh] = np.max(S)

plt.pcolormesh(t, f, G,cmap='Greys')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Interference Signals')
plt.colorbar()
plt.show()

Sblank = np.where(abs(abs(S) -  mean) >=  thresh, np.zeros(S.shape), S )
plt.pcolormesh(t, f, Sblank, cmap="jet")
#plt.imshow(Sblank)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Signal after blanking of RFI')
plt.colorbar()
plt.show()

Savg = np.where(abs(abs(S) -  mean) >=  thresh, mean*np.ones(S.shape), S )
plt.pcolormesh(t, f, Sblank, cmap="jet")
#plt.imshow(Sblank)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Signal after replacing contaminated data with local average')
plt.colorbar()
plt.show()
'''
