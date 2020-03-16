# RF and Optical Communication Course Project
# Credits : Yash Bhusare (23yashbhusare@gmail.com)
# Code to process the data-set and create the dynamic spectrum

import numpy as np
from scipy import signal,fftpack
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool 
import sys
from scipy.ndimage.interpolation import shift

# bandwidth 16.5 MHz, DM 0, Central Frequency 159 MHz

# program usage: python <exec> file_name time(1000) frequency(256) intregation point(60) central_freq BW DM(0)  0/1 (0  for img and 1 for numpy array)  

M_range = int(sys.argv[2])

a1 = np.zeros((int(sys.argv[3])-1,int(M_range)))
a2 = np.zeros((int(sys.argv[3])-1,int(M_range)))
a3 = np.zeros((int(sys.argv[3])-1,int(M_range)))
a4 = np.zeros((int(sys.argv[3])-1,int(M_range)))

# function to process the data in parallel and create dynamic spectrum
def dynamic_spectrum(m):
        z1 = np.empty(int(sys.argv[3])*2)
        z2 = np.empty(int(sys.argv[3])*2)
        f =  open(sys.argv[1] ,'rb') 
        M = m*int(sys.argv[4])*int(sys.argv[3])*4 +m*32
        f.seek(M)
        I_1 = np.zeros(int(sys.argv[3])-1)
        Q_1 = np.zeros(int(sys.argv[3])-1)
        U_1 = np.zeros(int(sys.argv[3])-1)
        for j in range(int(sys.argv[4])):
            f.read(32)
            infile = np.array(list(f.read(int(sys.argv[3])*4)))
            z1 = infile[::2]
            z2 = infile[1:][::2]
            temp = np.where(z1>128)
            z1[temp] = z1[temp] - 256
            temp = np.where(z2>128)
            z2[temp] = z2[temp] - 256
            del temp 
            fx = fftpack.fft(z1)[1:int(sys.argv[3])]            
            fy = fftpack.fft(z2)[1:int(sys.argv[3])]
            f1 = abs(fx)**2
            f2 = abs(fy)**2
            I = f1
            Q = f2
            t  = abs(fx*np.conj(fy))**2
            del fx 
            del fy
	    # I_1, Q_1,U_1 are just name dont get confused with stokes para
            I_1 = I_1 + I
            Q_1 = Q_1 + Q
            U_1 = U_1 + t
        c = np.array([m])
        r = np.concatenate((c,I_1,Q_1,U_1))
        print(r)
        return r
        del I_1,Q_1,U_1
        
start_time = time.time()

if __name__ == "__main__":
  p = Pool()
  for j in range(30):

    M_range1 = (np.arange(j*int(sys.argv[2]),(j+1)*int(sys.argv[2]),1))   # type total devide by 2 
       
    result = np.array( p.map( dynamic_spectrum, M_range1 ) )

    for i in range(1000):
        a1[:,int(result[i][0]) - j*int(sys.argv[2]) ] = result[i][1:int(sys.argv[3])]
        a2[:,int(result[i][0]) - j*int(sys.argv[2]) ] = result[i][int(sys.argv[3]):int(sys.argv[3])+int(sys.argv[3])-1]
        a3[:,int(result[i][0]) - j*int(sys.argv[2]) ] = result[i][int(sys.argv[3])+(int(sys.argv[3])-1):]
    
    # If non-zero dispersion measure, then perform dedispersion
    if int(sys.argv[7]) != 0 :
        I_1 = np.mean(a1,axis=-1)
        Q_1 = np.mean(a2,axis=-1)
        Q_2 = np.mean(a3,axis=-1)
        BW = float(sys.argv[6])
        central_freqency = float(sys.argv[5])
        DM = int(sys.argv[7])
        freq = np.linspace(central_freqency-BW/2,central_freqency+BW/2,int(sys.argv[3])-1)
        for i in range(int(sys.argv[3])-1):
            delay = 4.15*1000*DM*(((freq[i])**-2)-((central_freqency)**-2))   #calculating delay 
            a1[i] = a1[i]-I_1[i]                                              # subtracting mean for normalization 
            a2[i] = a2[i]-Q_1[i]
            a3[i] = a3[i]-Q_2[i]
            a4[i] = shift(a3[i] ,delay, cval = 0)  

    
    print("--- %s seconds ---" % (time.time() - start_time))#
    # Code to plot and save spectrograms, correlations
    plt.title("X_pol")
    plt.xlabel("Time")
    plt.ylabel("frequency measure")
    plt.imshow(a1,cmap="hot",aspect='auto')
    if sys.argv[8]:
        if sys.argv[8] == "0":
            plt.savefig("X_pol_.png")   
    plt.show()
    plt.title("Y_pol")
    plt.xlabel("Time")
    plt.ylabel("frequency measure")
    plt.imshow(a2,cmap="hot",aspect='auto')
    if sys.argv[8] == "0":
       plt.savefig("Y_pol_.png")
    plt.show()
    plt.title("Correlation")
    plt.xlabel("Time")
    plt.ylabel("frequency measure")
    plt.imshow(a3,cmap="hot",aspect='auto')
    if sys.argv[8]:
        if sys.argv[8] == "0":
            plt.savefig("Correlation_.png")
    plt.show()
    plt.title("DM corrected")
    plt.xlabel("Time points")
    plt.ylabel("frequency measure")
    plt.imshow((a4),cmap="hot",aspect='auto')
    if sys.argv[8]:
        if sys.argv[8] == "0" and sys.argv[5] != 0:
            plt.savefig("DM_corrected_.png")
    plt.show()
    plt.title("Time Stamped")
    plt.xlabel("Time points")
    plt.ylabel("Intensity")
    plt.plot(np.sum(a4,axis=0))
    if sys.argv[8]:
        if sys.argv[8] == "0" and sys.argv[5] != 0:
            plt.savefig("Time_Stamped_.png")
    plt.show()
    
    if sys.argv[8]:
        if sys.argv[8] == "1":
            np.save("X_pol_"+str(j) ,a1)
            np.save("Y_pol_"+str(j)  ,a2)
            np.save("Correlation_"+str(j) ,a3)
            np.save("Intensity_,"+str(j), np.sum(a4,axis=0))
            if sys.argv[5] != 0:
               np.save("DM_corrected_"+str(j) ,a4)     
  p.close()
  p.join()
   
