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
