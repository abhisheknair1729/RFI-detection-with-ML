import numpy as np
import binascii 

datta = []

with open("../data/ch03_CAS-A-IMG_20171203_151710_001.mbr", "rb" ) as f:
    data = f.read(32)

print(binascii.hexlify(data[22:24]))
