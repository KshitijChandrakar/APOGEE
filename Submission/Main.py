# %%
from astroquery.mast import Observations

import astropy.io.fits as fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

import scipy.stats as stats


import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os as os



# %%
path = "/home/asus/Downloads/APOGEE/"
StarHorsefile = "APOGEE_DR17_EDR3_STARHORSE_v2.fits"
APOGEEfile = "allStar-dr17-synspec_rev1.fits"
SHdatapath = path + StarHorsefile
APOGEEdatapath = path + APOGEEfile
SH_hdul = fits.open(SHdatapath)[1]
APOGEE_hdul = fits.open(APOGEEdatapath)[1]
# %%
N = APOGEE_hdul.data.shape[0]
N = 100000
N
# %%
def makedf(data, cols, N):
    df = pd.DataFrame()
    for i in cols:
        df[i] = pd.Series(data.data[i][:N])

    return df


# %% Numerical Columns with low Correlation
cols = ['J', 'H', 'K', 'SNREV', 'VHELIO_AVG', 'VSCATTER', 'RV_TEFF', 'RV_LOGG',
'RV_FEH', 'RV_ALPHA', 'RV_CARB', 'RV_CHI2', 'RV_CCFWHM', 'RV_AUTOFWHM',
'MEANFIB', 'SIGFIB', 'MIN_H', 'MAX_H', 'MIN_JK', 'MAX_JK',
'GAIAEDR3_PARALLAX', 'GAIAEDR3_PMRA', 'GAIAEDR3_PMDEC',
'GAIAEDR3_PHOT_G_MEAN_MAG', 'GAIAEDR3_PHOT_BP_MEAN_MAG',
'GAIAEDR3_PHOT_RP_MEAN_MAG', 'GAIAEDR3_DR2_RADIAL_VELOCITY',
'GAIAEDR3_R_MED_GEO', 'GAIAEDR3_R_LO_GEO', 'GAIAEDR3_R_HI_GEO',
'GAIAEDR3_R_MED_PHOTOGEO', 'GAIAEDR3_R_LO_PHOTOGEO',
'GAIAEDR3_R_HI_PHOTOGEO', 'ASPCAP_CHI2', 'FRAC_BADPIX', 'FRAC_LOWSNR',
'FRAC_SIGSKY', 'TEFF', 'LOGG', 'M_H', 'ALPHA_M', 'VMICRO', 'VMACRO',
'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE',
'SI_FE', 'P_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE', 'V_FE',
'CR_FE', 'MN_FE', 'FE_H', 'CO_FE', 'NI_FE', 'CU_FE', 'CE_FE', 'YB_FE',
'RA', 'DEC', 'GLON', 'GLAT']
# %%
df = makedf(APOGEE_hdul, cols, N)
# %%
HALO_BIT = 20
df["HALO_MEMBER"] = pd.Series(APOGEE_hdul.data['APOGEE2_TARGET2'][:N] & (1 << HALO_BIT)).astype(bool)

# %%
for col in df.columns:
    if df[col].dtype.kind in ['f', 'i']:  # float or integer
        df[col] = df[col].astype('float64')  # or 'int64'

# %%
threshold = len(df) * 0.90
cols_to_drop = df.columns[df.isna().sum() > threshold]
cols_to_drop
df = df.drop(columns=cols_to_drop)

# %%
df = df.dropna(subset=['RA'])
df = df.dropna(subset=['FE_H'])
df = df.dropna(subset=['GAIAEDR3_DR2_RADIAL_VELOCITY'])
df = df.dropna(subset=['GAIAEDR3_R_MED_GEO'])
df = df.dropna(subset=['K'])
df = df.dropna(subset=['CE_FE'])
df = df.dropna(subset=['NA_FE'])


# %%
df = df.dropna()
# %%
df.shape

# %%
with pd.option_context('display.max_rows', None):
    print(df.isna().sum())

# %%
cols_to_drop
''' Dropped Columns because of high na
Index(['GAIAEDR3_DR2_RADIAL_VELOCITY', 'M_H', 'ALPHA_M', 'VSINI', 'C_FE',
       'CI_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'P_FE',
       'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE', 'V_FE', 'CR_FE', 'MN_FE',
       'FE_H', 'CO_FE', 'NI_FE', 'CU_FE', 'CE_FE', 'YB_FE'],
      dtype='object')
'''
# %%
na_counts = df.isna().sum()
print(na_counts)
# %%
df.columns

# %%
HALO_MEMBER = df["HALO_MEMBER"].copy()
df = df.drop(columns=["HALO_MEMBER"])
# %%
X_train = df
Y_train = HALO_MEMBER
