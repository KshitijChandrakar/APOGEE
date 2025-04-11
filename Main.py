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

# %%
SH_hdul = fits.open(SHdatapath)[1]
APOGEE_hdul = fits.open(APOGEEdatapath)[1]


# %%
def makedf(data, cols, N):
    df = pd.DataFrame()
    for i in cols:
        df[i] = pd.Series(data.data[i][:N])

    return df


# %%
# cols = ["APOGEE_ID", "GLON", "GLAT", "RA", "DEC"]
# SH = pd.DataFrame(SH_hdul.data)

# %%
# cols = ["APOGEE_ID", "FE_H", "LOGG", "TEFF", "GLON", "GLAT", "RA", "DEC"]
# APOGEE = makedf(APOGEE_hdul, Text_Params, 10)

# %%
# APOGEE[Text_Params[1]]

# %% markdown
# ----------------------------------------------------------------------------

# %% code
GroupByParameter = "M_H"
Text_Params = ["APOGEE_ID", "ALT_ID", "LOGG", "FE_H", "VMICRO", "VMACRO","VHELIO_AVG", "TEFF"]
X_Param, Y_Param, Z_Param = "GLAT", "GLON", "GAIAEDR3_DR2_RADIAL_VELOCITY"
N = 100000
title = (
    X_Param + " vs " + Y_Param + " vs " + Z_Param + " - " + GroupByParameter
    if Z_Param != None
    else X_Param + " vs " + Y_Param + " - " + GroupByParameter
)
data = APOGEE_hdul.data

# %% markdown

# %%
print("Starting plot")
# %%
def Plot2D():
    im = plt.scatter(
        x=data[X_Param][:N],
        y=data[Y_Param][:N],
        c=data[GroupByParameter][:N],
        marker=".",
        s=1,
        zorder=1,
        cmap="jet",
        vmin=-1,
        vmax=0.5,
    )
    plt.colorbar(im, location="bottom", label=GroupByParameter)

    plt.xlabel(X_Param)
    plt.ylabel(Y_Param)
    plt.title(title)
    plt.savefig(path + title)
# %%
def Plot3D():
    textdf = makedf(APOGEE_hdul, Text_Params, N)
    fig = go.Figure(
        data=go.Scatter3d(
            x=data[X_Param][:N],
            y=data[Y_Param][:N],
            z=data[Z_Param][:N],
            customdata=textdf,
            mode="markers",
            hovertemplate=(
                str.join("",[Text_Params[i] + ": %{customdata[" + str(i) +"]}<br>" for i in range(len(Text_Params))]) +
                X_Param + ": %{x}<br>" +
                Y_Param + ": %{y}<br>" +
                Z_Param + ": %{z}<br>" +
                GroupByParameter + ": %{marker.color}<br>" +
                "<extra></extra>"
            ),
            marker=dict(
                size=3,
                opacity=0.3,
                color=data[GroupByParameter][:N],
                colorscale="Viridis",
                colorbar=dict(title=GroupByParameter),
                showscale=True,
            ),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=X_Param, yaxis_title=Y_Param, zaxis_title=Z_Param),
        width=800,
        height=600,
    )

    fig.write_html(path + title + ".html")  # Basic save

# %%
try:
    if Z_Param == None:
        Plot2D()
    else:
        Plot3D()
except KeyboardInterrupt:
    print("hie")
    exit(1)
