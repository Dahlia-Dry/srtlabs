# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits
hdul = fits.open('srt-azel-test/az.fits')
data = hdul[0].data
print(hdul[1].header['METADATA'])
az = []
el = []
for i in range(1,len(hdul)):
    metadata = hdul[i].header['METADATA'].split(',')
    data = data + hdul[i].data
    for element in metadata:
        if "motor_el" in element:
            el.append(float(element.split(':')[-1]))
        elif "motor_az" in element:
            az.append(float(element.split(':')[-1]))
print(az, el)
