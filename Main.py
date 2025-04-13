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
def makedf(data, cols, N):
    df = pd.DataFrame()
    for i in cols:
        df[i] = pd.Series(data.data[i][:N])

    return df


# %% Numerical Columns
cols = ['J', 'H', 'K', 'SNREV', 'VHELIO_AVG', 'VSCATTER', 'RV_TEFF', 'RV_LOGG', 'RV_FEH', 'RV_ALPHA', 'RV_CARB', 'RV_CHI2', 'RV_CCFWHM', 'RV_AUTOFWHM', 'MEANFIB', 'SIGFIB', 'MIN_H', 'MAX_H', 'MIN_JK', 'MAX_JK', 'GAIAEDR3_PARALLAX', 'GAIAEDR3_PMRA', 'GAIAEDR3_PMDEC', 'GAIAEDR3_PHOT_G_MEAN_MAG', 'GAIAEDR3_PHOT_BP_MEAN_MAG', 'GAIAEDR3_PHOT_RP_MEAN_MAG', 'GAIAEDR3_DR2_RADIAL_VELOCITY', 'GAIAEDR3_R_MED_GEO', 'GAIAEDR3_R_LO_GEO', 'GAIAEDR3_R_HI_GEO', 'GAIAEDR3_R_MED_PHOTOGEO', 'GAIAEDR3_R_LO_PHOTOGEO', 'GAIAEDR3_R_HI_PHOTOGEO', 'ASPCAP_CHI2', 'FRAC_BADPIX', 'FRAC_LOWSNR', 'FRAC_SIGSKY', 'TEFF', 'LOGG', 'M_H', 'ALPHA_M', 'VMICRO', 'VMACRO', 'VSINI', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'P_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE', 'V_FE', 'CR_FE', 'MN_FE', 'FE_H', 'CO_FE', 'NI_FE', 'CU_FE', 'CE_FE', 'YB_FE', 'RA', 'DEC', 'GLON', 'GLAT']
# %%
df = makedf(APOGEE_hdul, cols, 100000)

# %%
for col in df.columns:
    if df[col].dtype.kind in ['f', 'i']:  # float or integer
        df[col] = df[col].astype('float64')  # or 'int64'
# %%
df.dropna(axis=1,  thresh=5)
# %%
corr_matrix = df.corr()
plot = sns.heatmap(corr_matrix, annot=False)
plt.savefig(path + "CorrelationMatrixBetweenNumerical1.png",bbox_inches="tight")

# %%
corr_matrix
# %%
threshold = 0.9

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# %%

to_drop = [column for column in corr_matrix.columns if any(upper[column] > threshold)]
df.drop(columns=to_drop)

# %%

'''
Data Columns with dropped columns over 0.9 Correlation
['J', 'H', 'K', 'SNREV', 'VHELIO_AVG', 'VSCATTER', 'RV_TEFF', 'RV_LOGG',
'RV_FEH', 'RV_ALPHA', 'RV_CARB', 'RV_CHI2', 'RV_CCFWHM', 'RV_AUTOFWHM',
'MEANFIB', 'SIGFIB', 'MIN_H', 'MAX_H', 'MIN_JK', 'MAX_JK',
'GAIAEDR3_PARALLAX', 'GAIAEDR3_PMRA', 'GAIAEDR3_PMDEC',
'GAIAEDR3_PHOT_G_MEAN_MAG', 'GAIAEDR3_PHOT_BP_MEAN_MAG',
'GAIAEDR3_PHOT_RP_MEAN_MAG', 'GAIAEDR3_DR2_RADIAL_VELOCITY',
'GAIAEDR3_R_MED_GEO', 'GAIAEDR3_R_LO_GEO', 'GAIAEDR3_R_HI_GEO',
'GAIAEDR3_R_MED_PHOTOGEO', 'GAIAEDR3_R_LO_PHOTOGEO',
'GAIAEDR3_R_HI_PHOTOGEO', 'ASPCAP_CHI2', 'FRAC_BADPIX', 'FRAC_LOWSNR',
'FRAC_SIGSKY', 'TEFF', 'LOGG', 'M_H', 'ALPHA_M', 'VMICRO', 'VMACRO',
'VSINI', 'C_FE', 'CI_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE',
'SI_FE', 'P_FE', 'S_FE', 'K_FE', 'CA_FE', 'TI_FE', 'TIII_FE', 'V_FE',
'CR_FE', 'MN_FE', 'FE_H', 'CO_FE', 'NI_FE', 'CU_FE', 'CE_FE', 'YB_FE',
'RA', 'DEC', 'GLON', 'GLAT']
'''
# %%

try:
    if Z_Param == None:
        pass
        # Plot2D()
    else:
        pass
        # Plot3D()
except KeyboardInterrupt:
    print("hie")
    exit(1)
