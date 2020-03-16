# Radio Frequency Interference Detection and Mitigation using Machine Learning
# RF and Optical Engineering Course Project
# Abhishek Nair, Mohit Shrivastava, Mohammed Khandwawala

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# load x polarisation and y polarisation data
# Each array is a 2D spectrogram
data_x = np.load( "../data/X_pol_19.npy" )
data_y = np.load( "../data/Y_pol_19.npy" )

data_x = data_x[:,:256]
data_y = data_y[:,:256]

# obtain root mean square array
data = np.sqrt( (np.multiply( data_x, data_x ) + np.multiply( data_y, data_y ) )/2 )

# collect data statistics for each frequency channel
data_stats = []
for i in range(255):
    data_stats.append( stats.describe(data[i]) )

# array to store the mean of each frequency channel
mean = [ data_stats[i][2] for i in range(255) ]
avg_mean = np.mean(mean)
std_dev = np.std(mean)
freq_channels = np.linspace(151.75, 167.25, 255)
interference_freq_channels = freq_channels[np.where(mean - avg_mean > 2*std_dev)]
print( "Frequencies contaminated with RFI: " )
for f in interference_freq_channels:
    print(f)
# array to store the sum of all frequency components for each time step
# not very discriminative
# sum_val = [ np.sum(abs(data.T[i])) for i in range(1000) ]

# plotting the arrays
# plt.plot(sum_val, 'bo')
# plt.show()
plt.plot( mean, 'bo-')
plt.plot((avg_mean + 2*std_dev)*np.ones(len(mean)) )
plt.legend(["signal", "decision_threshold"])
plt.xlabel(" Normalized Frequencies (0 - 256) ")
plt.ylabel(" Mean Intensity Values ")
plt.title(" Mean Intensity vs Frequency " )
plt.show()

plt.imshow(data)
plt.xlabel(" Time ");
plt.ylabel(" Frequency ")
plt.title("Spectrogram of original data ")
plt.show()

data1 = np.where( data - avg_mean > 2*std_dev , np.zeros(data.shape), data)

plt.imshow(data1)
plt.xlabel(" Time ");
plt.ylabel(" Frequency ")
plt.title("Spectrogram of data after blanking RFI ")
plt.show()

data2 = np.where( data -avg_mean > 2*std_dev, avg_mean*np.ones(data.shape), data )
plt.imshow(data2)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("Spectrogram of data after replacing the contaminated data with local mean")
plt.show()

