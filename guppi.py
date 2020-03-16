
# coding: utf-8

# In[25]:
import sys

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftshift
from scipy.signal import spectrogram
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import kurtosis

N = 256
D = 100

# In[18]:


data = np.load("chan26.npy")

chan_1 = np.zeros(len(data))
chan_1 = np.transpose(data)[3]

print( kurtosis(chan_1, fisher=False))
#sys.exit(0)
plt.hist(chan_1,bins=np.linspace(min(chan_1), max(chan_1), 100))
plt.show()
sys.exit()

f_chan_1 = np.zeros( (D, N), dtype=complex )
i = 0
for j in range(D):
    f_chan_1[j] = fft(chan_1[ i : i + N ])
    i += int(N/2)

f_chan_1 = fftshift(f_chan_1)
f_power = abs(f_chan_1)**2

fig, ax = plt.subplots(1, 1)

ax.pcolormesh( f_power, cmap="coolwarm", antialiased=False )


# Smoothing Filter to remove noise but not RFI

window = np.hanning(N)
plt.plot(window)


a,b,c = spectrogram( chan_1, fs = 1.0, window=window, noverlap=int(0.75*N), nfft=N, return_onesided=False )
print(c.shape)

fig1, ax1 = plt.subplots(1, 1)
ax1.pcolormesh( fftshift(np.transpose(c)[:100]), cmap="coolwarm", antialiased=False )
plt.title("Spectrogram")
plt.show()

'''
fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.linspace(0, 8, n_power.shape[0])
Y = np.linspace(0, 512, n_power.shape[1])

X, Y = np.meshgrid(Y, X)

surf = ax.plot_surface(X, Y, n_power, cmap="coolwarm", linewidth=0, antialiased=False)
plt.show()




n_spectrum = np.sum(abs(n_f)**2, axis=0)
s_spectrum = np.sum(abs(s_f)**2, axis=0)


# In[ ]:


plt.plot (range(256),n_spectrum[:256])
plt.show()


# In[ ]:


plt.plot (range(256), s_spectrum[:256])
plt.show()
'''
