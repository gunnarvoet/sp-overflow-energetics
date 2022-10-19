# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# # %matplotlib notebook
# %matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import mkdir
import gsw
import xarray as xr
import gvpy as gv

import nslib as nsl

# %reload_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
cfg = nsl.io.load_config()

# %%
cfg.model.input.setup_files.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# # Generate mitgcm input data

# %% [markdown]
# Note: Endianness of the binary files written here is not explicitly set and will depend on the machine you are working on. Use [numpy.dtype.newbyteorder](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.newbyteorder.html) to explicitly set the endianness or compile mitgcm such that the machine default will be used.

# %% [markdown]
# It seems like float64 is the default data type in numpy, but let's be clear about this and write a little test/conversion function.

# %%
def CheckFloat64(x):
    if x.dtype == np.float64:
        print('its a float64')
    else:
        print('converting to float64')
        x = x.astype(np.float64)
    return x


# %% [markdown]
# Basic geometry for this case is 100 vertical levels and 8*250 horizontal levels, running on 8 cores.

# %%
nx = 1
ny = 3000
nz = 130

# %% [markdown]
# ## Horizontal resolution

# %%
a = np.linspace(1000, 50, 500, dtype=np.float64)
b = np.squeeze(np.ones((1, 2000), dtype=np.float64))*20
c = np.linspace(50, 1000, 500, dtype=np.float64)

dy = np.hstack((a,b,c))
# unlike Matlab cumsum np.cumsum preserves input shape
y  = np.cumsum(dy)
y  = y - y[700]

# make sure dy is a float64
dy = CheckFloat64(dy)
# save to binary file
with open(cfg.model.input.setup_files.joinpath("delYvar"), 'wb') as f:
    dy.tofile(f)

# %%
fig, ax = plt.subplots(2,1, sharex=True, figsize=(6,5))
ax[0].plot(dy, 'k.')
ax[0].set_title('cell size dy')
ax[0].set_ylabel('cell size [m]')

ax[1].plot(y/1000, 'k.')
ax[1].set_title('grid spacing y')
ax[1].set_ylabel('distance [km]')
ax[1].set_xlabel('grid index')

plt.tight_layout()

# %% [markdown]
# ## Vertical resolution

# %%
# from bottom up - start out with 20m resolution here
dz1 = np.ones(60)*20
rema = 5300 - np.sum(dz1)
xx = np.arange(1,71,1)+20
dz2 = xx*rema/np.sum(xx)
dz = np.hstack((dz1, dz2))
dz = np.flipud(dz)
z = np.cumsum(dz)
# make sure dz is in float64
dz = CheckFloat64(dz)
# save to binary file
with open(cfg.model.input.setup_files.joinpath("delZvar"), 'wb') as f:
    dz.tofile(f)

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(dz, -z, 'k.')
ax.set_ylabel('z [m]')
ax.set_xlabel('dz [m]')
plt.tight_layout()

# %% [markdown]
# ## Bottom topography

# %%
track = gv.io.loadmat(cfg.model.input.setup_files.joinpath('track_towyo_long.mat'))

# %%
from scipy.interpolate import interp1d
x = track['dist']
depth = -track['depth']
f = interp1d(x*1000, depth, bounds_error=False)
d2 = f(y)
d2[0:504] = -5082
d2[2524:] = d2[2524]

# make sure it's float64
d2 = CheckFloat64(d2)
# save to binary
with open(cfg.model.input.setup_files.joinpath("topogSamoa.bin"), 'wb') as f:
    d2.tofile(f)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)

ax[0].plot(y/1000,d2,'k')
ax[0].set_ylabel('z [m]')

ax[1].plot(y/1000, d2,'k');
ax[1].set(xlim=(-10,40))
for axi in ax:
    gv.plot.axstyle(axi)
    axi.set_xlabel('y [km]')

# %% [markdown]
# Adjust upstream botton depth to go to the same level as downstream.

# %%
y[504]/1e3

# %%
y[206]/1e3

# %%
f = interp1d(y[[206, 504]], [d2[2524], d2[505]])

# %%
test = f(y[206:505])

# %%
plt.plot(y[206:505]/1000, test, 'k.')

# %%
d2old = d2.copy()
d2[206:505] = test
d2[0:206] = test[0]

# %%
# make sure it's float64
d2 = CheckFloat64(d2)
# save to binary
with open(cfg.model.input.setup_files.joinpath("topogSamoa.bin"), 'wb') as f:
    d2.tofile(f)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)

ax[0].plot(y/1000,d2,'k')
ax[0].set_ylabel('z [m]')

ax[1].plot(y/1000, d2,'k');
ax[1].set(xlim=(-10,40))
for axi in ax:
    gv.plot.axstyle(axi)
    axi.set_xlabel('y [km]')
# plt.savefig('fig/topo.pdf')

# %% [markdown]
# Make sure the flat bottom connects nicely to the real bathymetry

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)
ax[0].plot(d2,'k');
ax[0].set(xlim=(300, 700))
ax[1].plot(d2,'k');
ax[1].set(xlim=(2400, 2600))
for axi in ax:
    gv.plot.axstyle(axi)

# %% [markdown]
# See how steppy the topography is at 20m vertical resolution (although I realize this doesn't make sense to look at since there are partially filled bottom cells).

# %%
xx = d2.copy()
bins = -z
inds = np.digitize(xx, bins)
d2d = -z[inds]

# %%
fig, ax = plt.subplots(figsize=(8,3))
[ax.axhline(y=yy, linewidth=0.2) for yy in np.arange(-5300,-4700,20)]
ax.plot(y/1000, d2d, 'k.')
ax.set_ylabel('z [m]')
ax.set_xlabel('y [km]')
ax.set_xlim([-110,110])
plt.tight_layout()
# plt.savefig('fig/topo_resolution.pdf')

# %% [markdown]
# Find grid point of sill crest for stratification interpolation further below.

# %%
d2i = np.argmax(d2)
print(d2i)
print(d2[d2i])

# %% [markdown]
# ## Stratification
# Using a linear equation of state with $\alpha_T=2\times10^{-4}$.

# %%
gravity = 9.81;
talpha = 2.0e-4;

# %% [markdown]
# Load ctd profiles

# %%
c = nsl.io.load_ctd_profiles()

# %%
# drop profile with bad longitude
c = c.where(c.lon<-160, drop=True)

# %%
sec9 = c.where(c.section==9, drop=True)

# %%
b = gv.ocean.smith_sandwell(c.lon.data, c.lat.data)
b.plot(cmap='Blues_r')
plt.plot(sec9.lon, sec9.lat, 'ko')
tmp1 = sec9.th.where(sec9.station==12, drop=True)
tmp2 = sec9.th.where(sec9.station==19, drop=True)

plt.plot(tmp1.lon, tmp1.lat, 'ro')
plt.plot(tmp2.lon, tmp2.lat, 'yo')

# %% [markdown]
# ok, let's use stations 12 and 19

# %%
tmp = sec9.th.where(sec9.station==12, drop=True)
plt.plot(tmp, tmp.z, 'b')
tmp = sec9.th.where(sec9.station==19, drop=True)
plt.plot(tmp, tmp.z, 'r')
plt.gca().set(ylim=(5000, 3500), xlim=(0.67,1.25))

# %% [markdown]
# Calculate N$^2$ from the CTD data, smooth it and then translate it to the linear equation of state.
# $$
# N^2 = -\frac{g}{\rho} \frac{d \rho}{dz}
# $$
# $$
# \rho = \alpha_T \theta
# $$
# $$
# T_{z0} = \frac{N^2}{g \alpha_T}
# $$

# %%
tmp = sec9.where(sec9.station==12, drop=True).squeeze()
tmp['s'] = tmp.s.interpolate_na(dim='z')
SA = gsw.SA_from_SP(tmp.s.squeeze(), tmp.p.squeeze(), tmp.lon, tmp.lat)
CT = gsw.CT_from_t(SA, tmp.t.squeeze(), tmp.p.squeeze())
N2, pmid = gsw.Nsquared(SA, CT, tmp.p.squeeze(), tmp.lat)
ctdz = tmp.z[1:]
good = np.where(~np.isnan(tmp.t.squeeze()))
from scipy.signal import convolve2d
# convolve2d wants 2D arrays
N2smooth = convolve2d(np.reshape(N2[good],(len(N2[good]),1)), np.reshape(np.ones(100)/100,(100,1)), mode='same')
# back to 1D array
N2smooth = np.squeeze(N2smooth)
f = interp1d(ctdz[good], N2smooth, bounds_error=False)
N2 = f(z)
# translate N2 to the linear equation of state
tz0 = N2/(gravity*talpha)
# integrate vertically
t = np.cumsum(-tz0*dz)
TrefS = t-t[0]+18.353
# replace bottom nan's with deepest value
ind = np.where(~np.isnan(TrefS))[0]
first, last = ind[0], ind[-1]
TrefS[last + 1:] = TrefS[last]
NrefS = N2

# %%
tmp = sec9.where(sec9.station==19, drop=True).squeeze()
SA = gsw.SA_from_SP(tmp.s.squeeze(), tmp.p.squeeze(), tmp.lon, tmp.lat)
CT = gsw.CT_from_t(SA, tmp.t.squeeze(), tmp.p.squeeze())
N2, pmid = gsw.Nsquared(SA, CT, tmp.p.squeeze(), tmp.lat)
ctdz = tmp.z[1:]
good = np.where(~np.isnan(tmp.t.squeeze()))
from scipy.signal import convolve2d
# convolve2d wants 2D arrays
N2smooth = convolve2d(np.reshape(N2[good],(len(N2[good]),1)), np.reshape(np.ones(100)/100,(100,1)), mode='same')
# back to 1D array
N2smooth = np.squeeze(N2smooth)
f = interp1d(ctdz[good], N2smooth, bounds_error=False)
N2 = f(z)
# translate N2 to the linear equation of state
tz0 = N2/(gravity*talpha)
# integrate vertically
t = np.cumsum(-tz0*dz)
TrefN = t-t[0]+19.294
# replace bottom nan's with deepest value
ind = np.where(~np.isnan(TrefN))[0]
first, last = ind[0], ind[-1]
TrefN[last + 1:] = TrefN[last]
NrefN = N2

# %%
fig, ax = plt.subplots(1,3, figsize=(9,4))
ax[0].plot(NrefS,z)
ax[0].plot(NrefN,z)
ax[0].set(ylim=(5500,2000), xlim=(-3e-7,7e-6))

ax[1].plot(TrefS,z)
ax[1].plot(TrefN,z)
ax[1].set(ylim=(5500,2000), xlim=(0.5, 2))

ax[2].plot(TrefS,z)
ax[2].plot(TrefN,z)
ax[2].set(ylim=(1000, 0), xlim=(2, 20))

# %%
tmp = sec9.where(sec9.station==12, drop=True)
print(tmp.th.min())
# print(np.nanmin(ctd1['theta1']))
print(np.nanmin(TrefS))

# %%
fig, ax = plt.subplots(1,4,sharey=True, figsize=(8,4))
ax[0].plot(N2,z)
ax[0].set_title('N$^2$')
ax[1].plot(tz0,z)
ax[1].set_title('T$_{Z0}$')
ax[2].plot(t,z)
ax[2].set_title('t')
ax[3].plot(TrefS,z)
ax[3].set_title('t$_{ref}$')
ax[0].invert_yaxis()
for axi in ax:
    axi.grid()
plt.tight_layout()
# plt.savefig('fig/n2-tref.pdf')

# %%
fig, ax = plt.subplots(1,2)
ax[0].plot(TrefN,z)
ax[0].plot(TrefS,z)
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(TrefN,z)
ax[1].plot(TrefS,z)
ax[1].invert_yaxis()
ax[1].grid()
ax[1].set_ylim(5300,4000)
ax[1].set_xlim(0.5,1.5)

# %% [markdown]
# Make the profiles the same in the upper layer

# %%
TrefN[z<4200] = TrefS[z<4200]

# %%
fig, ax = plt.subplots(1,2)
ax[0].plot(TrefN,z)
ax[0].plot(TrefS,z)
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(TrefN,z)
ax[1].plot(TrefS,z)
ax[1].invert_yaxis()
ax[1].grid()
ax[1].set_ylim(5300,4000)
ax[1].set_xlim(0.5,1.5)

plt.tight_layout()
# plt.savefig('fig/tref-profiles.pdf')

# %% [markdown]
# Save profiles for open boundary conditions

# %%
TrefN = CheckFloat64(TrefN)
# save to binary
with open(cfg.model.input.setup_files.joinpath("OB_North_T.bin"), 'wb') as f:
    TrefN.tofile(f)
TrefS = CheckFloat64(TrefS)
# save to binary
with open(cfg.model.input.setup_files.joinpath("OB_South_T.bin"), 'wb') as f:
    TrefS.tofile(f)

# %% [markdown]
# also save the southern profile as reference profile

# %%
with open(cfg.model.input.setup_files.joinpath("Tref"), 'wb') as f:
    TrefS.tofile(f)

# %%
TrefN.shape

# %% [markdown]
# ## Generate inital stratification

# %%
T = np.zeros((nz, ny))
for i, (ts, tn) in enumerate(zip(TrefS, TrefN)):
    # d2i is the index at the ridge crest
    f = interp1d(y[[0, d2i, 2600, 2999]], [ts, ts, tn, tn], bounds_error=False)
    T[i,:] = f(y)

# convert to 3D array (not sure if needed for 2D field, but nice to have for future cases)
# Tinit = np.zeros([nx,ny,nz])
# for k in np.arange(0,nx):
#     Tinit[k,:,:] = np.transpose(T[:,0:ny])

# this seems to work (not sure why it has to be nx, nz, ny)
Tinit2 = np.zeros([nx,nz,ny])
for k in np.arange(0,nx):
    Tinit2[k,:,:] = T[:,0:ny]

# Tinit = CheckFloat64(Tinit)
# # save to binary
# with open("T.init", 'wb') as f:
#     Tinit.tofile(f)
    
Tinit2 = CheckFloat64(Tinit2)
# save to binary
with open(cfg.model.input.setup_files.joinpath("T.init"), 'wb') as f:
    Tinit2.tofile(f)

# %%
Tinit2.shape

# %%
fig, ax = plt.subplots()
h = ax.pcolormesh(Tinit2[0,:,:], vmin=0.6, vmax=1.1)
ax.contour(Tinit2[0,:,:], levels=np.arange(0.6,0.79,0.01), colors='k')
plt.colorbar(h)

# %%
fig, ax = plt.subplots()
cs = plt.contourf(y/1000, z, np.ma.masked_invalid(T), levels=[0.6, 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,2,3,4,5,10,15,20], cmap='Spectral_r')
for c in cs.collections:
    c.set_edgecolor('face')
plt.contour(y/1000, z, np.ma.masked_invalid(T), levels=np.arange(0.6,0.8,0.01), colors='k', alpha=0.3)
plt.gca().invert_yaxis()
plt.colorbar(cs, label='T$_{init}$')
plt.plot(y/1000, -d2, 'k');
plt.xlabel('y [km]')
plt.ylabel('depth [m]')
plt.tight_layout()
# plt.savefig('fig/Tinit.pdf')

# %%
fig, ax = gv.plot.newfig()
cs = plt.contourf(y/1000, z, np.ma.masked_invalid(T),
                  levels=np.arange(0.65,1.5,0.05),
                  cmap='Spectral_r')
for c in cs.collections:
    c.set_edgecolor('face')
ax.contour(y/1000, z, np.ma.masked_invalid(T), levels=np.arange(0.6,1.5,0.05), colors='k', alpha=0.3)
ax.invert_yaxis()
plt.colorbar(cs, label='T$_{init}$')
ax.fill_between(y/1000, np.abs(d2), np.ones_like(d2)*1e4, color='k');
ax.set(xlabel='y [km]', ylabel='depth [m]', xlim=(-25, 65), ylim=(5300, 3500))
ax.grid(False)
plt.tight_layout()
# plt.savefig('fig/Tinit_zoom.pdf')

# %% [markdown]
# # Plot bathy and resolution in y and z

# %%
fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2,rowspan=2)
ax2 = plt.subplot2grid((3,3), (2,0), colspan=2,sharex=ax1)
ax3 = plt.subplot2grid((3,3), (0,2), rowspan=2,sharey=ax1)

fig.subplots_adjust(wspace=0.1,hspace=0.1) 

# ax1.plot(y/1000,-d2,'k')
ax1.set(ylim=(5500,0),ylabel='Depth [m]')
cs = ax1.contour(y/1000, z, np.ma.masked_invalid(T),
                  levels=np.arange(0.65,25,0.05), colors='k', alpha=0.5, linewidths=0.25)

ax1.fill_between(y/1000, np.abs(d2), np.ones_like(d2)*1e4, color='k');

ax2.plot(y/1000,dy,'k')
ax2.set(ylabel=r'$\Delta\,\mathrm{y}$ [m]',xlabel='y [km]')

ax3.plot(dz, z, 'k')
ax3.set(xlim=(0,100),xlabel=r'$\Delta\,\mathrm{z}$ [m]')

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

import string
for n, axi in enumerate([ax1,ax2,ax3]):
    if n in [2]:
        axi.text(0.02, 0.92, string.ascii_lowercase[n], transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')
    else:
        axi.text(0.02, 0.08, string.ascii_lowercase[n], transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')
    

for axi in [ax2,ax3]:
    axi.grid()

# plt.savefig('fig/bathy_and_resolution_new.pdf',bbox_inches='tight')
# gv.plot.png('model_setup')

# %%
l1 = np.arange(0.6, 1.25, 0.05)
len1 = len(l1)
l2 = np.arange(1.2, 20, 1)
len2 = len(l2)
ll = np.hstack([l1, l2])

# %%
len1

# %%
len2

# %%
fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2,rowspan=2)
ax2 = plt.subplot2grid((3,3), (2,0), colspan=2,sharex=ax1)
ax3 = plt.subplot2grid((3,3), (0,2), rowspan=2,sharey=ax1)
cax = plt.subplot2grid((3,3), (2,2))

fig.subplots_adjust(wspace=0.08,hspace=0.1) 

ax1.set(ylim=(5500,0),ylabel='Depth [m]')

cs = ax1.contourf(y/1000, z, np.ma.masked_invalid(T), levels=l1,
                  cmap='YlGnBu_r')
cb = plt.colorbar(cs, cax=cax, orientation='horizontal')
pos = cax.get_position()
cax.set_position([pos.x0, pos.y0+pos.height/2, pos.width/2, pos.height/10])
cb.set_ticks([0.6, 0.8, 1.0, 1.2])
cb.set_label(r'$\theta$ [°C]')

cs2 = ax1.contourf(y/1000, z, np.ma.masked_invalid(T), levels=l2,
                  cmap='YlOrRd')
for c in cs2.collections:
    # Fix for the white lines between contour levels
    c.set_edgecolor("face")
    # Rasterized contours in pdfs
    c.set_rasterized(True)

cax2 = fig.add_axes(cax.get_position())
cb2 = plt.colorbar(cs2, cax=cax2, orientation='horizontal')
pos2 = cax2.get_position()
cax2.set_position([pos2.x0+pos2.width, pos2.y0, pos2.width, pos2.height])
cb2.set_ticks([5, 10, 15])

ax1.fill_between(y/1000, np.abs(d2), np.ones_like(d2)*1e4, color='k');

# c1 = ax1.contour(y/1000, z, np.ma.masked_invalid(T), levels=[0.78],
#                   colors='w', linewidths=1)

c2 = ax1.contour(y/1000, z, np.ma.masked_invalid(T), levels=[1.2],
                  colors='k', alpha=0.5, linewidths=1, linestyles='--')
plt.clabel(c2, fmt='%1.1f°', inline_spacing=1, manual=[(250, 3700)])

ax2.plot(y/1000,dy,'k')
ax2.set(ylabel=r'$\Delta\,\mathrm{y}$ [m]',xlabel='y [km]')

ax3.plot(dz, z, 'k')
ax3.set(xlim=(0,100),xlabel=r'$\Delta\,\mathrm{z}$ [m]')

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

import string
for n, axi in enumerate([ax1,ax2,ax3]):
    if n in [0, 2]:
        axi.text(0.02, 0.92, string.ascii_lowercase[n], transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')
    else:
        axi.text(0.02, 0.08, string.ascii_lowercase[n], transform=axi.transAxes,
              horizontalalignment='left', fontweight='bold')    

for axi in [ax2,ax3]:
    axi.grid()

nsl.io.save_png('model_setup')
nsl.io.save_pdf('model_setup')

# %%
