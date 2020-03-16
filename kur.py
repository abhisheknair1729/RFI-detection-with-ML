# RF and Optical Communication Course Project
# Abhishek Nair, Mohit Shrivastava , Mohammed Khandwawala
# Code to compute kurtosis of raw data
# Result: 1.14 ( should be 3 )

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as mpl

f = open("../data/ch03_CAS-A-IMG_20171203_151710_001.mbr", "rb" )

integration_time = 60
freq_channel = 256
avg_kur = 0
count = 0

for m in range(30):
	M = m*integration_time*freq_channel*4 +m*32
	f.seek(M)
	for step in range(integration_time):
		f.read(32)
		data_arr = np.array(list(f.read(freq_channel*4)))
		kur = stats.kurtosis(data_arr,axis = None)
		avg_kur += kur
		count += 1

avg_kur = avg_kur/count
mpl.plot(data_arr)
mpl.show()
print("Avg kur: {}".format(avg_kur+3))
