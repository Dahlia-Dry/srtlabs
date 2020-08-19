# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits
hdul = fits.open('g80.fits')
data = hdul[0].data
print(hdul[0].header)
for i in range(1,len(hdul)):
    data = data + hdul[i].data

freqs = np.fft.fftfreq(256,2000000)
plt.scatter(freqs,data,s=0.3)
plt.title('g70')
plt.show()
