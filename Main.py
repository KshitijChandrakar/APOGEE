
# %%
%matplotlib widget
# %%
from astroquery.mast import Observations

import astropy.io.fits as fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

import scipy.stats as stats

import matplotlib.pyplot as plt
import numpy as np

import plotly.express as px
import plotly.offline as pyo

import seaborn as sns
import pandas as pd
import os as os

# %%
path = '/home/asus/Downloads/APOGEE/'
StarHorsefile = 'APOGEE_DR17_EDR3_STARHORSE_v2.fits'
APOGEEfile='allStar-dr17-synspec_rev1.fits'
SHdatapath = path + StarHorsefile
APOGEEdatapath = path + APOGEEfile

# %%
SH_hdul = pd.DataFrame(fits.open(SHdatapath)[1].data)
# %%
SH_hdul.shape

# %%
tempAPOGEE = fits.open(APOGEEdatapath)

# %%

APOGEE_hdul = pd.DataFrame(tempAPOGEE[1].data)

# %%
merged = pd.merge(APOGEE, SH, on='APOGEE_ID', how='inner')

# %%
GroupByParameter = "FE_H"
X_Param, Y_Param, Z_Param = "VMICRO", "VHELIO_AVG", None
N = 100000
title = X_Param + " vs " + Y_Param + " vs " + Z_Param + " - " + GroupByParameter if Z_Param != None else X_Param + " vs " + Y_Param + " - " + GroupByParameter

# %%
im = plt.scatter(x = data[X_Param][:N], y = data[Y_Param][:N], c=data[GroupByParameter][:N],
                 marker='.', s=1, zorder=1,
                 cmap='jet', vmin=-1, vmax=0.5)
plt.colorbar(im, location='bottom', label='Metallicity' + GroupByParameter)

plt.xlabel(X_Param)
plt.ylabel(Y_Param)
plt.title(title)
plt.savefig(path + title)
