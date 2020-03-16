
# coding: utf-8

# In[25]:


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftshift
from mpl_toolkits.mplot3d import Axes3D


# In[18]:


data = []
with open("data_1", "r") as data_file:
    data = data_file.readlines()[:512*8]


# In[19]:


print(len(data))
print( data[0] )
print( [ int(k) for k in data[0].split(" ") ] )


# In[20]:


north = np.zeros(len(data), dtype=int)
south = np.zeros(len(data), dtype=int)

for i in range(len(data)):
    data[i] = [int(k) for k in data[i].split(" ")]
    data[i] = np.array(data[i])
    north[i] = data[i][0]
    south[i] = data[i][1]


# In[21]:


print( type(north) )
print(north[0])


# In[22]:


n_f = np.zeros((int(len(data)/512), 512), dtype=complex)
s_f = np.zeros((int(len(data)/512), 512), dtype=complex)

for i in range(8):
    n_f[i] = fft(north[i:i+512])
    s_f[i] = fft(south[i:i+512])

n_power = abs(n_f)**2
s_power = abs(s_f)**2


# In[31]:


fig, ax = plt.subplots(1, 1)

ax.pcolormesh( n_power, cmap="coolwarm", antialiased=False )
plt.show()

# In[43]:


fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.linspace(0, 8, n_power.shape[0])
Y = np.linspace(0, 512, n_power.shape[1])

X, Y = np.meshgrid(Y, X)

surf = ax.plot_surface(X, Y, n_power, cmap="coolwarm", linewidth=0, antialiased=False)
plt.show()

# In[ ]:

'''
n_spectrum = np.sum(abs(n_f)**2, axis=0)
s_spectrum = np.sum(abs(s_f)**2, axis=0)


# In[ ]:


plt.plot (range(256),n_spectrum[:256])
plt.show()


# In[ ]:


plt.plot (range(256), s_spectrum[:256])
plt.show()
'''
