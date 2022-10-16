#!/usr/bin/env python
# coding: utf-8
"""
Towyo calculations.
"""

import numpy as np
import xarray as xr
import gsw
import scipy as sp
import gvpy as gv
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.signal as sg
from scipy.integrate import cumtrapz
from typing import Type
from tqdm import tqdm  # a little progress bar
import box


@xr.register_dataarray_accessor("ta")
class TowyoVar(object):
    """
    This class adds methods to xarray dataarrays under `ta`.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def plot(self, **kwargs):
        """Plot towyo field.

        Returns
        -------
        h :
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        h = gv.figure.pcm(self._obj.dist, self._obj.z, self._obj, ax=ax, **kwargs)
        ax.set(ylim=(5300, 4000))
        plt.colorbar(h)
        return h


@xr.register_dataset_accessor("ty")
class Towyo(object):
    """
    This class adds methods to xarray datasets. Methods are accessible under
    `ty`.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        self._topo = None
        self._epsot = None
        self._dens = None
        self._boundary_layer_mask = None
        self._fill_bottom_values = None

    # properties can be called without parentheses
    @property
    def print_temp(self):
        """Print first value of temperature field."""
        print(self._obj["t"].values[0][0])

    @property
    def print_mean_temp(self):
        print(self._obj["t"].mean().values)

    @property
    def print_max_temp(self):
        print(self._obj["t"].max().values)

    def AddTopo(self):
        """Add bathymetry to towyo dataset.

        Notes
        -----
        From the 200m resolution merged product of Smith & Sandwell and multibeam.

        Adds variables `dist` and `topo` to the dataset.
        """
        if self._topo is None:
            print("adding topo...")
            topopath = Path("/Users/gunnar/Projects/sp/data/gridded/mb/")
            bathyfile = topopath / "merged_200_-171.5_-167.5_-11_-6.5.nc"
            bathy = xr.open_dataset(bathyfile)
            out = gv.ocean.bathy_section(
                bathy, self._obj.lon.data, self._obj.lat.data, 0.1, 10
            )
            # Approximate offset of the package behind the ship is about 850m.
            TopoOffset = 0.85
            dist = out["odist"]
            f = sp.interpolate.interp1d(out["idist"] + TopoOffset, out["itopo"])
            topo = f(dist)
            self._obj.coords["dist"] = ("x", dist)
            self._obj.dist.attrs["name"] = "Distance"
            self._obj.dist.attrs["units"] = "km"
            self._obj["topo"] = (("x"), topo)
            self._obj.topo.attrs["name"] = "Bottom Depth"
            self._obj.topo.attrs["units"] = "m"
            self._topo = "done"
        else:
            print("bathymetry already added!")

    def AddEpsOverturns(self):
        """Add turbulent dissipation based on Thorpe scales to towyo dataset.

        Notes
        -----
        Adds variables `eps`, `k`, `Lt` and `otn2` to the dataset.
        """
        if self._epsot is None:
            print("overturn calculations...")
            eps = np.zeros_like(self._obj.t.values) * np.nan
            k = np.zeros_like(self._obj.t.values) * np.nan
            Lt = np.zeros_like(self._obj.t.values) * np.nan
            n2 = np.zeros_like(self._obj.t.values) * np.nan

            ni = np.arange(0, len(self._obj.lon.values), 1)

            for j in ni:
                out = gv.ocean.eps_overturn(
                    self._obj.p.values[:, j],
                    self._obj.z.values,
                    self._obj.t.values[:, j],
                    self._obj.s.values[:, j],
                    self._obj.lon.values[j],
                    self._obj.lat.values[j],
                )
                eps[:, j] = out["eps"]
                k[:, j] = out["k"]
                Lt[:, j] = out["Lt"]
                n2[:, j] = out["n2"]

            self._obj["eps"] = (["z", "x"], eps)
            self._obj["k"] = (["z", "x"], k)
            self._obj["Lt"] = (["z", "x"], Lt)
            self._obj["otn2"] = (["z", "x"], n2)
            self._epsot = "done"
        else:
            print("overturn calculations already done!")

    def AddDensity(self):
        if self._dens is None:
            print("density calcs...")
            # SA
            self._obj["SA"] = (
                ["z", "x"],
                gsw.SA_from_SP(
                    self._obj.s, self._obj.p, self._obj.lon, self._obj.lat
                ).data,
            )
            # CT
            self._obj["CT"] = (
                ["z", "x"],
                gsw.CT_from_t(self._obj.SA, self._obj.t, self._obj.p).data,
            )
            # potential temperature
            self._obj["PT"] = (
                ["z", "x"],
                gsw.pt0_from_t(self._obj.SA, self._obj.t, self._obj.p).data,
            )
            # rho
            self._obj["gsw_rho"] = (
                ["z", "x"],
                gsw.rho(self._obj.SA, self._obj.CT, self._obj.p).data,
            )
            # sigma4
            self._obj["gsw_sigma4"] = (
                ["z", "x"],
                gsw.pot_rho_t_exact(self._obj.SA, self._obj.t, self._obj.p, 4000).data
                - 1000,
            )
            # N^2
            N2p, Np = gsw.Nsquared(
                self._obj.SA, self._obj.CT, self._obj.p, self._obj.lat
            )
            # interpolate N2 from Np to p
            N2 = np.zeros_like(self._obj.t) * np.nan
            for i, (N2i, Npi, pi) in enumerate(zip(N2p.T, Np.T, self._obj.p.T)):
                f = sp.interpolate.interp1d(Npi, N2i, bounds_error=False)
                N2[:, i] = f(pi)
            self._obj["N2"] = (["z", "x"], N2)
            self._dens = "done"
        else:
            print("density already done")

    def AddBathyMask(self):
        """Add a bathymetry mask to the Dataset with 1 above bottom and np.nan below.

        Notes
        -----
        Adds variable `bathymask`."""
        bathymask = xr.ones_like(self._obj.t)
        self._obj["bathymask"] = bathymask.where(bathymask.z < self._obj.topo, np.nan)

    def BoundaryLayerMask(self):
        """Add boundary layer masks for CTD and velocity.

        Adds the variables `BoundaryLayerMaskCTD` and `BoundaryLayerMaskVel`.
        """

        if self._boundary_layer_mask is None:
            print("generating boundary layer mask...")
            ztm, zvm = _BottomDistance(self._obj)
            self._obj["BottomLayerStartCTD"] = (["x"], ztm + 1)
            self._obj["BottomLayerStartVel"] = (["x"], zvm + 1)
            blctd = np.ones_like(self._obj.v) * np.nan
            blvel = np.ones_like(self._obj.v) * np.nan
            for i, (ztmi, topoi) in enumerate(zip(ztm, self._obj.topo.values)):
                z1 = gv.misc.nearidx(self._obj.z.values, ztmi)
                z2 = gv.misc.nearidx(self._obj.z.values, topoi)
                blctd[z1 + 1 : z2, i] = 1
            for i, (zvmi, topoi) in enumerate(zip(zvm, self._obj.topo.values)):
                z1 = gv.misc.nearidx(self._obj.z.values, zvmi)
                z2 = gv.misc.nearidx(self._obj.z.values, topoi)
                blvel[z1 + 1 : z2, i] = 1
            self._obj["BoundaryLayerMaskCTD"] = (["z", "x"], blctd)
            self._obj["BoundaryLayerMaskVel"] = (["z", "x"], blvel)
            self._boundary_layer_mask = "done"

    def FillBottomValuesDensity(self):
        """Fill the bottom layer with constant density values.

        Adds the variable `sg4_filled`.
        """

        if self._fill_bottom_values is None:
            print("filling bottom layer with constant density...")
            # depth matrix
            Z = np.tile(self._obj.z, (len(self._obj.dist), 1))
            Z = Z.T
            # find indices of deepest observation
            Zt = np.ma.masked_where(np.isnan(self._obj.t.values), Z)
            ztmi = np.nanargmax(Zt, axis=0)
            # gather densities at deepest observation in each profile
            sigmas = []
            for i, zi in enumerate(ztmi):
                sigmas.append(self._obj.gsw_sigma4[zi, i].values)
            sigmas = np.array(sigmas)
            ii = np.argwhere(self._obj.bdist.values == 0)
            for i in ii:
                if self._obj.bdist[i - 1] > 0 and i > 0:
                    print(
                        "replacing values at {:1.1f} km with left neighbor".format(
                            self._obj.dist.values[i][0]
                        )
                    )
                    sigmas[i] = sigmas[i - 1]
                elif self._obj.bdist[i + 1] > 0 and i < len(sigmas):
                    print(
                        "replacing values at {:1.1f} km with right neighbor".format(
                            self._obj.dist.values[i][0]
                        )
                    )
                    sigmas[i] = sigmas[i + 1]
            # replace sigmas where bdist==0 with neighboring sigmas
            out = self._obj.BoundaryLayerMaskCTD * sigmas
            mask = self._obj.BoundaryLayerMaskCTD == 1
            self._obj["sg4_filled"] = (
                ["z", "x"],
                self._obj.gsw_sigma4.copy().data,
            )
            self._obj.sg4_filled.values[mask] = out.values[mask]
            self._fill_bottom_values = "done"

    def CalculateSpeedDirection(self):
        """Calculate speed and direction from u and v.

        Notes
        -----
        Adds variables `speed` and `direction` to the Dataset.
        """
        V = self._obj.u + 1j * self._obj.v
        self._obj["speed"] = (["z", "x"], np.absolute(V).data)
        self._obj["direction"] = (["z", "x"], np.angle(V, deg=True).data)

    def HorizontalGradients(self):
        """Calculate horizontal gradients in temperature and density.

        Notes
        -----
        Adds the variables `HorTempGrad` and `HorDensGrad` to the Dataset.
        """
        HorTempGrad = np.gradient(self._obj.PT, self._obj.dist * 1000, axis=1)
        self._obj["HorTempGrad"] = (["z", "x"], HorTempGrad)
        self._obj.HorTempGrad.attrs["name"] = "grad(T)"
        self._obj.HorTempGrad.attrs["units"] = "°C/m"
        HorDensGrad = np.gradient(self._obj.gsw_sigma4, self._obj.dist * 1000, axis=1)
        self._obj["HorDensGrad"] = (["z", "x"], HorDensGrad)
        self._obj.HorDensGrad.attrs["name"] = "grad(rho)"
        self._obj.HorDensGrad.attrs["units"] = r"kg/m$^4$"

    def KineticEnergy(self):
        """Calculate kinetic energy.

        Notes
        -----
        Adds variable `KE` to Dataset.
        """
        self._obj["KE"] = (["z", "x"], (0.5 * self._obj.v**2 * 1046).data)
        self._obj.KE.attrs["name"] = "Kinetic Energy"
        self._obj.KE.attrs["units"] = r"J/m$^3$"

    def AvailablePotentialEnergy(self, ctd):
        """Calculate available potential energy.

        Parameters
        ----------
        ctd : xarray.Dataset or xarray.DataArray
            Reference CTD profile either from downstream measurements or from
            sorted initial model density.

        Notes
        -----
        Adds variable `APE` (for downstream reference) or `APEs` (for sorted
        reference profile) to Dataset.
        """
        # Determine which type of reference we are dealing with.
        if ctd.attrs["title"] == "downstream_density_ctd_profile":
            ref_model = False
        else:
            ref_model = True
        if ref_model:
            print("APE based on sorted density")
        else:
            print("APE based on downstream density")

        if ref_model:
            ffsg = ctd.data
            ffz = -1 * ctd.Z.data
        else:
            ffsg = ctd.sg4filt.data
            ffz = -1 * ctd.z.data  # z increases upwards
        LAPE = self._obj.gsw_sigma4.data * np.nan
        for i, sg4 in tqdm(enumerate(self._obj.gsw_sigma4.values.T)):
            # make z increase upwards (same as in model analysis)
            zz = -1 * self._obj.z.values
            for j, zipitems in enumerate(zip(sg4, zz)):
                r = zipitems[0]
                z = zipitems[1]
                if ~np.isnan(r):
                    if r > np.max(ffsg):
                        i2 = gv.misc.nearidx(ffz, -5300)
                        # index of far field depth == towyo depth
                        iz = gv.misc.nearidx(ffz, z)
                    else:
                        # index of far field sg==towyo sigma
                        i2 = gv.misc.nearidx(ffsg, r)
                        # index of far field depth == towyo depth
                        iz = gv.misc.nearidx(ffz, z)
                    if iz < i2:
                        H = -1 * (ffz[iz:i2] - z)
                        drho = r - ffsg[iz:i2]
                    else:
                        H = -1 * (ffz[i2:iz] - z)
                        drho = r - ffsg[i2:iz]
                    LAPE[j, i] = gsw.grav(
                        np.nanmean(self._obj.lat), 0
                    ) * sp.integrate.trapz(drho, H)
        if ref_model:
            self._obj["APEs"] = (["z", "x"], LAPE)
            self._obj.APEs.attrs["name"] = "APE"
            self._obj.APEs.attrs["units"] = r"J/m$^3$"
        else:
            self._obj["APE"] = (["z", "x"], LAPE)
            self._obj.APE.attrs["name"] = "APE"
            self._obj.APE.attrs["units"] = r"J/m$^3$"

    def KE_flux(self):
        """Calculate the flux of kinetic energy.

        Notes
        -----
        Kinetic energy flux $\mathbf{F}_\mathrm{KE}$ is calculated as
        $$\mathbf{F}_\mathrm{KE} = \mathbf{v} E_K$$

        or in this case only in one dimension as $$F_\mathrm{KE} = v E_K$$

        Adds variable `KEflux` to the Dataset.
        """
        self._obj["KEflux"] = (["z", "x"], (self._obj.v * self._obj.KE).data)

    def APE_flux(self):
        """Calculate the flux of available potential energy.

        Notes
        -----
        One-dimensional flux of available potential energy is calculated
        as $$F_\mathrm{PE} = v E_P$$

        Adds variable `APEflux` to the dataset.
        """
        self._obj["APEflux"] = (["z", "x"], (self._obj.v * self._obj.APE).data)
        self._obj["APEsflux"] = (
            ["z", "x"],
            (self._obj.v * self._obj.APEs).data,
        )

    def MeanDensityProfile(self):
        """Calculate a mean density profile.

        Notes
        -----
        Adds variables `mean_rho` and `mean_rho_lp` to Dataset.
        """
        # Design the Buterworth filter
        N = 3  # Filter order
        Wn = 1 / 200.0  # Cutoff frequency
        B, A = sg.butter(N, Wn, output="ba")
        # Calculate mean density and filter it
        mean_rho_tmp = np.nanmean(self._obj.gsw_sigma4 + 1000, 1)
        self._obj["mean_rho"] = ("z", mean_rho_tmp)
        mean_rho_lp = sg.filtfilt(B, A, mean_rho_tmp[~np.isnan(mean_rho_tmp)])
        mean_rho = np.zeros_like(mean_rho_tmp) * np.nan
        mean_rho[~np.isnan(mean_rho_tmp)] = mean_rho_lp
        self._obj["mean_rho_lp"] = (["z"], mean_rho)

    def WindowedMeanDensityProfile(self, deltax: float = 2.5):
        """Calculate mean density profiles using a sliding window.

        Parameters
        ----------
        deltax : float
            +/- this value in km for the window length

        Notes
        -----
        Adds `mean_rho_windowed_lp` to Dataset.
        """
        # First, design the Buterworth filter
        N = 3  # Filter order
        Wn = 1 / 200.0  # Cutoff frequency/2
        B, A = sg.butter(N, Wn, output="ba")
        mean_rho_windowed = np.zeros_like(self._obj.gsw_sigma4) * np.nan
        for ii, x in enumerate(self._obj.dist.values):
            xi = np.where((self._obj.dist > x - deltax) & (self._obj.dist < x + deltax))
            rho_tmp = np.squeeze(self._obj.gsw_sigma4.values[:, xi] + 1000)
            rho_tmp_mean = np.nanmean(rho_tmp, 1)
            nn = np.where(~np.isnan(rho_tmp_mean))
            rho_tmp_mean_lp = sg.filtfilt(B, A, rho_tmp_mean[nn])
            mean_rho_windowed[nn, ii] = rho_tmp_mean_lp
        self._obj["mean_rho_windowed_lp"] = (["z", "x"], mean_rho_windowed)

    def DensityAnomaly(self):
        """Calculate the density anomaly as the difference of in-situ and mean
        density profiles.

        Notes
        -----
        Adds variables `rho_anom`, `rho_anom_fill` and `rho_anom_windowed` to
        Dataset.
        """
        rho_anom = np.transpose(
            np.subtract(
                self._obj.gsw_sigma4.values.T + 1000,
                self._obj.mean_rho_lp.values,
            )
        )
        # also for density filled all the way to the bottom
        rho_anom_fill = np.transpose(
            np.subtract(
                self._obj.sg4_filled.values.T + 1000,
                self._obj.mean_rho_lp.values,
            )
        )
        rho_anom_windowed = np.subtract(
            self._obj.gsw_sigma4 + 1000, self._obj.mean_rho_windowed_lp
        )
        self._obj["rho_anom"] = (["z", "x"], rho_anom.data)
        self._obj["rho_anom_fill"] = (["z", "x"], rho_anom_fill.data)
        self._obj["rho_anom_windowed"] = (["z", "x"], rho_anom_windowed.data)

    def DensityAnomalySortedReference(self, rhos):
        """Calculate the density anomaly as the difference of in-situ and
        reference profile based on the sorted initial model density field.

        Parameters
        ----------
        rhos : dict or xarray.DataArray
            Reference density profile from sorted initial model density. Needs
            to be in sg4-like units.

        Notes
        -----
        Adds variable `rho_anom_s` and `rho_anom_fill_s` to Dataset.
        """
        self._obj["rho_anom_s"] = self._obj.gsw_sigma4 - rhos
        self._obj["rho_anom_fill_s"] = self._obj.sg4_filled - rhos

    def PressureAnomaly(self, StartIndex):
        """
        Calculate the pressure anomaly from density anomalies.
        """
        p_anom_window = np.zeros_like(self._obj.PT) * np.nan
        p_anom_window[StartIndex:, :] = np.cumsum(
            self._obj.rho_anom_windowed.values[StartIndex:, :] * 9.81, 0
        )
        self._obj["p_anom_windowed"] = (["z", "x"], p_anom_window)
        # calculate a non-windowed pressure anomaly for form drag
        p_anom = np.zeros_like(self._obj.PT) * np.nan
        p_anom[StartIndex:, :] = np.cumsum(
            self._obj.rho_anom_fill.values[StartIndex:, :] * 9.81, 0
        )
        self._obj["p_anom"] = (["z", "x"], p_anom)
        # calculate pressure anomaly for form drag
        p_anom = np.zeros_like(self._obj.PT) * np.nan
        p_anom[StartIndex:, :] = np.cumsum(
            self._obj.rho_anom_fill_s.values[StartIndex:, :] * 9.81, 0
        )
        self._obj["p_anom_s"] = (["z", "x"], p_anom)

    def IsopycnalDisplacement(self):
        rho1 = self._obj.gsw_sigma4.values + 1000
        z1 = self._obj.z.values.astype("float")
        etatmp = np.zeros_like(rho1) * np.nan
        for ii, ri in enumerate(rho1.T):
            frho = sp.interpolate.interp1d(
                self._obj.mean_rho_windowed_lp[:, ii],
                z1,
                fill_value=np.nan,
                bounds_error=False,
            )
            zz = frho(ri)
            dz = zz - z1
            etatmp[:, ii] = dz
        self._obj["eta"] = (["z", "x"], etatmp)
        self._obj.eta.attrs["units"] = "m"
        self._obj.eta.attrs["name"] = r"$\eta$"

    def VerticalVelocityFromEta(self):
        u = 0.25
        x2 = np.diff(self._obj.dist) / 2 + self._obj.dist[0:-1]
        detadx = np.diff(self._obj.eta, 1) / np.diff(self._obj.dist * 1000)
        # with constantu
        wEtaCU = u * detadx
        weta = np.zeros_like(self._obj.t) * np.nan
        # interpolate deta/dx back to regular x grid and multiply with
        # observed alongstream velocity
        for i, dd in enumerate(detadx):
            f = sp.interpolate.interp1d(
                x2.values, dd, bounds_error=False, fill_value=np.nan
            )
            weta[i, :] = f(self._obj.dist.values)
        weta = weta * self._obj.v
        tmp = xr.DataArray(
            wEtaCU,
            coords={"z": self._obj.z.values, "x2": x2.values},
            dims=["z", "x2"],
        )
        self._obj["weta_const_v"] = tmp
        self._obj["weta"] = (["z", "x"], weta.data)
        # Not sure why z looses it's attributes but adding them back here.
        self._obj.z.attrs["name"] = "Depth"
        self._obj.z.attrs["units"] = "m"

    def MeanVelocities(self):
        N = 3  # Filter order
        Wn = 1 / 200.0  # Cutoff frequency
        B, A = sg.butter(N, Wn, output="ba")

        mean_w = np.nanmean(self._obj.w, 1)
        mean_ww = np.nanmean(self._obj.weta, 1)
        mean_u = np.nanmean(self._obj.u, 1)
        mean_v = np.nanmean(self._obj.v, 1)

        tmp = sg.filtfilt(B, A, mean_w[~np.isnan(mean_w)])
        mean_w_lp = np.zeros_like(mean_w) * np.nan
        mean_w_lp[~np.isnan(mean_w)] = tmp

        tmp = sg.filtfilt(B, A, mean_ww[~np.isnan(mean_ww)])
        mean_ww_lp = np.zeros_like(mean_ww) * np.nan
        mean_ww_lp[~np.isnan(mean_ww)] = tmp

        tmp = sg.filtfilt(B, A, mean_u[~np.isnan(mean_u)])
        mean_u_lp = np.zeros_like(mean_u) * np.nan
        mean_u_lp[~np.isnan(mean_u)] = tmp

        tmp = sg.filtfilt(B, A, mean_v[~np.isnan(mean_v)])
        mean_v_lp = np.zeros_like(mean_v) * np.nan
        mean_v_lp[~np.isnan(mean_v)] = tmp

        self._obj["mean_w"] = ("z", mean_w)
        self._obj["mean_w_lp"] = ("z", mean_w_lp)
        self._obj["mean_weta"] = ("z", mean_ww)
        self._obj["mean_weta_lp"] = ("z", mean_ww_lp)
        self._obj["mean_u"] = ("z", mean_u)
        self._obj["mean_u_lp"] = ("z", mean_u_lp)
        self._obj["mean_v"] = ("z", mean_v)
        self._obj["mean_v_lp"] = ("z", mean_v_lp)

    def WindowedMeanVelocities(self, deltax=2.5):
        # deltax = 2.5  # +/- this value in km for the window length
        # First, design the Buterworth filter
        N = 3  # Filter order
        Wn = 1 / 200.0  # Cutoff frequency/2
        B, A = sg.butter(N, Wn, output="ba")

        mean_w_windowed = np.zeros_like(self._obj.w) * np.nan
        for ii, x in enumerate(self._obj.dist.values):
            xi = np.where((self._obj.dist > x - deltax) & (self._obj.dist < x + deltax))
            w_tmp = np.squeeze(self._obj.w.values[:, xi])
            w_tmp_mean = np.nanmean(w_tmp, 1)
            nn = np.where(~np.isnan(w_tmp_mean))
            w_tmp_mean_lp = sg.filtfilt(B, A, w_tmp_mean[nn])
            mean_w_windowed[nn, ii] = w_tmp_mean_lp
        self._obj["mean_w_windowed_lp"] = (["z", "x"], mean_w_windowed)

        mean_weta_windowed = np.zeros_like(self._obj.weta) * np.nan
        for ii, x in enumerate(self._obj.dist.values):
            xi = np.where((self._obj.dist > x - deltax) & (self._obj.dist < x + deltax))
            weta_tmp = np.squeeze(self._obj.weta.values[:, xi])
            weta_tmp_mean = np.nanmean(weta_tmp, 1)
            nn = np.where(~np.isnan(weta_tmp_mean))
            weta_tmp_mean_lp = sg.filtfilt(B, A, weta_tmp_mean[nn])
            mean_weta_windowed[nn, ii] = weta_tmp_mean_lp
        self._obj["mean_weta_windowed_lp"] = (["z", "x"], mean_weta_windowed)

        mean_u_windowed = np.zeros_like(self._obj.u) * np.nan
        for ii, x in enumerate(self._obj.dist.values):
            xi = np.where((self._obj.dist > x - deltax) & (self._obj.dist < x + deltax))
            u_tmp = np.squeeze(self._obj.u.values[:, xi])
            u_tmp_mean = np.nanmean(u_tmp, 1)
            nn = np.where(~np.isnan(u_tmp_mean))
            u_tmp_mean_lp = sg.filtfilt(B, A, u_tmp_mean[nn])
            mean_u_windowed[nn, ii] = u_tmp_mean_lp
        self._obj["mean_u_windowed_lp"] = (["z", "x"], mean_u_windowed)

        mean_v_windowed = np.zeros_like(self._obj.v) * np.nan
        for ii, x in enumerate(self._obj.dist.values):
            xi = np.where((self._obj.dist > x - deltax) & (self._obj.dist < x + deltax))
            v_tmp = np.squeeze(self._obj.v.values[:, xi])
            v_tmp_mean = np.nanmean(v_tmp, 1)
            nn = np.where(~np.isnan(v_tmp_mean))
            v_tmp_mean_lp = sg.filtfilt(B, A, v_tmp_mean[nn])
            mean_v_windowed[nn, ii] = v_tmp_mean_lp
        self._obj["mean_v_windowed_lp"] = (["z", "x"], mean_v_windowed)

    def VelocityAnomalies(self):
        """Calculate velocity anomalies from windowed mean velocities"""
        self._obj["wp"] = self._obj.w - self._obj.mean_w_windowed_lp
        self._obj["wetap"] = self._obj.weta - self._obj.mean_weta_windowed_lp
        self._obj["up"] = self._obj.u - self._obj.mean_u_windowed_lp
        self._obj["vp"] = self._obj.v - self._obj.mean_v_windowed_lp

    def WaveEnergyFluxes(self):
        """Calculate small-scale internal wave energy fluxes and pressure work
        terms.
        """
        self._obj["wp_pp"] = self._obj.p_anom_windowed * self._obj.wp
        self._obj["wetap_pp"] = self._obj.p_anom_windowed * self._obj.wetap
        self._obj["vp_pp"] = self._obj.p_anom_windowed * self._obj.vp
        self._obj["PressureWorkHorizSorted"] = self._obj.p_anom_s * self._obj.v
        self._obj["PressureWorkHoriz"] = self._obj.p_anom * self._obj.v
        self._obj["PressureWorkVertSorted"] = self._obj.p_anom_s * self._obj.w
        self._obj["PressureWorkVert"] = self._obj.p_anom * self._obj.w

    def MomentumFluxes(self):
        r"""Calculate momentum flux

        Notes
        -----
        Momentum Flux is calculated as $\rho_0 w^\prime v^\prime$.

        Adds variable `vp_wp` to the Dataset.
        """
        rho0 = 1045
        self._obj["vp_wp"] = self._obj.vp * self._obj.wp * rho0

    def FillInBackgroundDissipation(self):
        """Fill in background dissipation of 1e-11 where no overturns detected.

        Adds variable `epsfilled` to Dataset.
        """
        self._obj["epsfilled"] = (
            self._obj.eps.where(~np.isnan(self._obj.eps), 1e-11) * self._obj.bathymask
        )

    def IntegrateDissipation(self, cfg: box.Box):
        r"""Integrate epsilon vertically over the lower layer
        and cumulatively in the horizontal.

        Parameters
        ----------
        cfg : box.Box
            Config read from `config.yml` via :class:`nslib.io.load_config`

        Notes
        -----
        $$\rho_0 \int \int_{bottom}^{interface} \epsilon\ dz\ dx$$

        Define upper integration limit in $\sigma_4$-coordinates in
        `config.yml` under `cfg.parameters.towyo.interfacesg4`.

        Adds variables `EpsVI` and `LowerLayerEpsIntegrated` to the Dataset.
        """
        interface = cfg.parameters.towyo.interfacesg4
        self._obj["EpsVI"] = (
            ["x"],
            self._obj.eps.where(self._obj.gsw_sigma4 > interface).sum(dim="z").data,
        )
        rho0 = self._obj.gsw_sigma4.mean().values + 1000
        self._obj["LowerLayerEpsIntegrated"] = (
            ["x"],
            rho0
            * sp.integrate.cumtrapz(
                self._obj.EpsVI, self._obj.dist * 1000, initial=0
            ).data,
        )

    def VerticalIntegralLowerLayer(self, field: str, interface: float = 45.94):
        """Integrate a field vertically over the lower layer.

        Parameters
        ----------
        field : str
            Key to field
        interface: float, optional
            Density interface (in sigma_4). Defaults to 45.94.

        Notes
        -----
        The result is saved as a new DataArray in the Dataset with 'VI'
        appended to the key.
        """
        out = field + "VI"
        self._obj[out] = (
            ["x"],
            self._obj[field].where(self._obj.gsw_sigma4 > interface).sum(dim="z").data,
        )

    def IntegrateAlongIsopycnal(self, field, distmin=0, distmax=25):
        outname = field + "_along_isopycnal"
        out = []
        delta_sg4 = 0.01
        tmp = self._obj.where((self._obj.dist < distmax) & (self._obj.dist > distmin))
        for sgi in self._obj.sg4bins.values:
            field_avg = (
                tmp[field]
                .where(
                    (tmp.gsw_sigma4 > sgi - delta_sg4)
                    & (tmp.gsw_sigma4 <= sgi + delta_sg4)
                )
                .mean(dim="z")
            )
            field_avg[np.isnan(field_avg)] = 0
            # now integrate along x
            out = np.append(out, sp.integrate.trapz(field_avg, x=tmp.dist * 1000))
        self._obj[outname] = (["sg4bins"], out)

    def AverageAlongIsopycnal(self, field):
        outname = field + "_avg_along_isopycnal"
        out = []
        delta_sg4 = 0.01
        tmp = self._obj.where(self._obj.dist < 25)
        for sgi in self._obj.sg4bins.values:
            field_avg = (
                tmp[field]
                .where(
                    (tmp.gsw_sigma4 > sgi - delta_sg4)
                    & (tmp.gsw_sigma4 <= sgi + delta_sg4)
                )
                .mean(dim=("z", "x"))
            )
            out = np.append(out, field_avg)
        self._obj[outname] = (["sg4bins"], out)

    def FormDrag(self, maxdist: float):
        r"""Calculate form drag across the ridge.

        Parameters
        ----------
        maxdist : float
            Will integrate along x from 0 to this distance in km.

        Notes
        -----
        Form drag is calculated here as
        $$D_F = -\int p^\prime \frac{db}{dx} dx$$
        where $\frac{db}{dx}$ is the bottom slope.
        """
        # bottom slope
        topo_gradient = np.gradient(-1 * self._obj.topo, self._obj.dist * 1000)
        self._obj["BottomSlope"] = (["x"], topo_gradient)
        # bottom pressure
        # first, repeat z vector along x
        Z = np.tile(self._obj.z.values, (len(self._obj.dist.values), 1)).T
        Zt = np.ma.masked_where(np.isnan(self._obj.sg4_filled.values), Z)
        ztmi = np.nanargmax(Zt, axis=0)
        BottomPressure = []
        for i, zi in enumerate(ztmi):
            BottomPressure.append(self._obj.p_anom_s[zi, i].values)
        self._obj["BottomPressure"] = (["x"], np.array(BottomPressure))
        FormDragTMP = -1 * (self._obj.BottomPressure) * self._obj.BottomSlope
        mask = self._obj.dist < maxdist
        self._obj["FormDrag"] = np.trapz(FormDragTMP[mask], self._obj.dist[mask] * 1000)
        self._obj["FormDragStress"] = self._obj.FormDrag / (maxdist * 1e3)
        print("Form Drag is {:1.0f} N/m".format(self._obj.FormDrag.values.tolist()))
        print(
            "Stress associated with form drag is {:1.2f} N/m^2".format(
                self._obj.FormDragStress.values.tolist()
            )
        )
        print(
            "Energy loss associated with form drag is {:1.1f} kW/m at 0.1 m/s upstream flow speed".format(
                (self._obj.FormDrag * 0.1 * 1e-3).values.tolist()
            )
        )

    def BottomDrag(self, cd: float = 2e-3):
        r"""
        Calculate bottom drag along the flow based on quadratic drag parameterization.

        Parameters
        ----------
        cd : float, optional
            Drag coefficient. Defaults to $2\times10^{-3}$.

        Notes
        -----
        Bottom drag is calculated here as
        $\tau_B = \rho C_D u^2$
        in units of [N/m$^2$].
        """
        rho = 1e3
        v = self._obj.v
        # find bottom-most velocity in each profile
        vbot = []
        for vi in v.T:
            xi = np.flatnonzero(~np.isnan(vi))
            vbot.append(vi[xi[-1]])
        vbot = np.array(vbot)
        taub = rho * cd * vbot**2
        self._obj["BottomDrag"] = (["x"], taub)
        self._obj.BottomDrag.attrs["name"] = "Bottom Drag"
        self._obj.BottomDrag.attrs["units"] = r"N/m$^2$"
        dp = taub * vbot
        self._obj["BottomDragDissipation"] = (["x"], dp)
        self._obj.BottomDragDissipation.attrs["name"] = "Bottom Drag Dissipation"
        self._obj.BottomDragDissipation.attrs["units"] = r"W/m$^2$"
        dpint = sp.integrate.cumtrapz(dp, self._obj.dist * 1e3, initial=0)
        self._obj["BottomDragDissipationIntegrated"] = (["x"], dpint)
        self._obj.BottomDragDissipationIntegrated.attrs[
            "name"
        ] = "Cumulatively Integrated Bottom Drag Dissipation"
        self._obj.BottomDragDissipationIntegrated.attrs["units"] = r"W/m"

    # plotting functions
    def PlotBottomLayerCTD(self, ax):
        ax.fill_between(
            self._obj.dist,
            self._obj.BottomLayerStartCTD + 1,
            self._obj.topo,
            color="k",
            alpha=0.2,
        )

    def PlotBottom(self, ax):
        """Plot bottom topography along towyo line.

        Parameters
        ----------
        ax : matplotlib.axes
            Axis for plotting
        """

        ax.fill_between(self._obj.dist, self._obj.topo, 10000, color="k", alpha=0.8)

    def PlotHorizontalTemperatureGradient(self, ax):
        """Plot horizontal temperature graidents.

        Parameters
        ----------
        ax : matplotlib.axes
            Axis for plotting.
        """

        ax.pcolormesh(
            self._obj.dist,
            self._obj.z,
            self._obj.HorTempGrad,
            cmap="RdBu_r",
            vmin=-1e-4,
            vmax=1e-4,
        )
        ax.set(ylim=(5300, 4000), xlabel="distance [km]", ylabel="depth [m]")
        self.PlotBottom(ax)

    def PlotVolumeFlux(self, ax):
        """Plot volume flux.

        Parameters
        ----------
        ax : matplotlib.axes
            Axis for plotting.
        """
        ax.plot(self._obj.dist, self._obj.vVI, color="0.5", linewidth=0.5)
        self._obj.vVI.rolling(x=20, center=True).mean().plot(ax=ax, x="dist")
        ax.set(xlabel="dist [km]", ylabel="depth [m]")


# Energy Budget Terms
def calculate_energy_budget_terms(ty, cfg):
    """Calculate energy budget terms for a towyo section.

    Parameters
    ----------
    ty : xr.Dataset
        Towyo section
    cfg : box.Box
        Config read from config.yml` via :class:`nslib.io.load_config`

    Returns
    -------
    budget_terms : xr.Dataset
        Energy budget terms.

    Notes
    -----
    Horizontal integration limits are set in `config.yml`.
    """
    year = np.datetime_as_string(ty.time.isel(x=0).data, unit="Y")
    # Read integration ranges from config file
    upstream = np.array(cfg.parameters.towyo.energy_budget_upstream_range)
    dnstream = np.array(cfg.parameters.towyo.energy_budget_dnstream_range)

    upstream_index = np.flatnonzero(
        (ty.dist > upstream.min()) & (ty.dist < upstream.max())
    )
    dnstream_index = np.flatnonzero(
        (ty.dist > dnstream.min()) & (ty.dist < dnstream.max())
    )

    apeflux_upstream = ty.APEfluxVI.sel(x=upstream_index).mean(dim="x")
    apeflux_upstream_std = ty.APEfluxVI.sel(x=upstream_index).std(dim="x")
    apeflux_dnstream = ty.APEfluxVI.sel(x=dnstream_index).mean(dim="x")
    apeflux_dnstream_std = ty.APEfluxVI.sel(x=dnstream_index).std(dim="x")

    apesflux_upstream = ty.APEsfluxVI.sel(x=upstream_index).mean(dim="x")
    apesflux_upstream_std = ty.APEsfluxVI.sel(x=upstream_index).std(dim="x")
    apesflux_dnstream = ty.APEsfluxVI.sel(x=dnstream_index).mean(dim="x")
    apesflux_dnstream_std = ty.APEsfluxVI.sel(x=dnstream_index).std(dim="x")

    keflux_upstream = ty.KEfluxVI.sel(x=upstream_index).mean(dim="x")
    keflux_upstream_std = ty.KEfluxVI.sel(x=upstream_index).std(dim="x")
    keflux_dnstream = ty.KEfluxVI.sel(x=dnstream_index).mean(dim="x")
    keflux_dnstream_std = ty.KEfluxVI.sel(x=dnstream_index).std(dim="x")

    apeflux_div = (apeflux_dnstream - apeflux_upstream).data
    apesflux_div = (apesflux_dnstream - apesflux_upstream).data
    keflux_div = (keflux_dnstream - keflux_upstream).data

    PressureWorkHoriz_upstream = ty.PressureWorkHorizVI.sel(x=upstream_index).mean(
        dim="x"
    )
    PressureWorkHoriz_upstream_std = ty.PressureWorkHorizVI.sel(x=upstream_index).std(
        dim="x"
    )
    PressureWorkHoriz_dnstream = ty.PressureWorkHorizVI.sel(x=dnstream_index).mean(
        dim="x"
    )
    PressureWorkHoriz_dnstream_std = ty.PressureWorkHorizVI.sel(x=dnstream_index).std(
        dim="x"
    )
    PressureWorkHoriz_div = (
        PressureWorkHoriz_dnstream - PressureWorkHoriz_upstream
    ).data

    PressureWorkHorizSorted_upstream = ty.PressureWorkHorizSortedVI.sel(
        x=upstream_index
    ).mean(dim="x")
    PressureWorkHorizSorted_dnstream = ty.PressureWorkHorizSortedVI.sel(
        x=dnstream_index
    ).mean(dim="x")
    PressureWorkHorizSorted_div = (
        PressureWorkHorizSorted_dnstream - PressureWorkHorizSorted_upstream
    ).data

    keflux_min = (keflux_dnstream - keflux_dnstream_std) - (
        keflux_upstream + keflux_upstream_std
    )
    keflux_max = (keflux_dnstream + keflux_dnstream_std) - (
        keflux_upstream - keflux_upstream_std
    )
    keflux_range = np.array([keflux_min.data, keflux_max.data])

    apeflux_min = (apeflux_dnstream - apeflux_dnstream_std) - (
        apeflux_upstream + apeflux_upstream_std
    )
    apeflux_max = (apeflux_dnstream + apeflux_dnstream_std) - (
        apeflux_upstream - apeflux_upstream_std
    )
    apeflux_range = np.array([apeflux_min.data, apeflux_max.data])

    apesflux_min = (apesflux_dnstream - apesflux_dnstream_std) - (
        apesflux_upstream + apesflux_upstream_std
    )
    apesflux_max = (apesflux_dnstream + apesflux_dnstream_std) - (
        apesflux_upstream - apesflux_upstream_std
    )
    apesflux_range = np.array([apesflux_min.data, apesflux_max.data])

    PressureWorkHoriz_min = (
        PressureWorkHoriz_dnstream - PressureWorkHoriz_dnstream_std
    ) - (PressureWorkHoriz_upstream + PressureWorkHoriz_upstream_std)
    PressureWorkHoriz_max = (
        PressureWorkHoriz_dnstream + PressureWorkHoriz_dnstream_std
    ) - (PressureWorkHoriz_upstream - PressureWorkHoriz_upstream_std)
    PressureWorkHoriz_range = np.array(
        [PressureWorkHoriz_min.data, PressureWorkHoriz_max.data]
    )

    # Turbulent dissipation is cumulative. Choose the most downstream
    # one.
    dissipation = ty.LowerLayerEpsIntegrated.sel(x=dnstream_index[-1]).data

    # TODO: move this into bottom drag calculation
    bottom_drag_dissipation_int = cumtrapz(
        ty.BottomDragDissipation.isel(x=ty.dist < 17).data,
        ty.dist.isel(x=ty.dist < 17).data * 1e3,
    )
    bottom_dissipation = np.array(bottom_drag_dissipation_int[-1])

    # TODO: need to calculate this for km 0 to 17 and km 5 to 12 for a range.
    vertical_wave_flux = ty.wp_pp_along_isopycnal.interp(
        sg4bins=cfg.parameters.towyo.interfacesg4
    ).data

    # vertical pressure work term
    vertical_pressure_work = ty.PressureWorkVert_along_isopycnal.interp(
        sg4bins=cfg.parameters.towyo.interfacesg4
    ).data
    vertical_pressure_work_sorted = ty.PressureWorkVertSorted_along_isopycnal.interp(
        sg4bins=cfg.parameters.towyo.interfacesg4
    ).data

    print("\n", year, "\n------")
    print("towyo APE flux div: {:1.1e} W/m".format(apeflux_div))
    print("towyo APEs flux div: {:1.1e} W/m".format(apesflux_div))
    print("towyo KE flux div: {:1.1e} W/m".format(keflux_div))
    print("towyo dissipation:", dissipation)
    print("bottom dissipation:", bottom_dissipation)
    print("towyo IW flux div:", vertical_wave_flux)
    print("towyo vert pressure work div:", vertical_pressure_work_sorted)
    print("towyo horiz pressure work div:", PressureWorkHorizSorted_div)
    print("towyo eps + IW flux:", vertical_wave_flux + dissipation)

    bt = xr.Dataset(
        dict(
            apeflux_div=apeflux_div,
            apesflux_div=apesflux_div,
            keflux_div=keflux_div,
            dissipation=dissipation,
            bottom_dissipation=bottom_dissipation,
            vertical_wave_flux=vertical_wave_flux,
            vertical_pressure_work=vertical_pressure_work,
            vertical_pressure_work_sorted=vertical_pressure_work_sorted,
            horizontal_pressure_work=PressureWorkHoriz_div,
            horizontal_pressure_work_sorted=PressureWorkHorizSorted_div,
            apeflux_range=(["range"], apeflux_range),
            apesflux_range=(["range"], apesflux_range),
            keflux_range=(["range"], keflux_range),
            PressureWorkHoriz_range=(["range"], PressureWorkHoriz_range),
        )
    )
    bt.apeflux_div.attrs = dict(long_name="APE flux divergence", units="W/m")
    bt.apesflux_div.attrs = dict(long_name="APE flux divergence", units="W/m")
    bt.keflux_div.attrs = dict(long_name="KE flux divergence", units="W/m")
    bt.dissipation.attrs = dict(long_name="epsilon integrated", units="W/m")
    bt.bottom_dissipation.attrs = dict(
        long_name="bottom dissipation integrated", units="W/m"
    )
    bt.vertical_wave_flux.attrs = dict(
        long_name="vertical wave flux integrated", units="W/m"
    )

    return bt


def calculate_matching_form_drag_velocity(a, bt, year):
    if year == 2012:
        ty = a["t12"]
    else:
        ty = a["t14"]
    bty = bt.sel(year=year)

    print("----")
    print(f"{year}")
    print("----\n")
    print("including vertical pressure work term:")
    loss_terms = (
        bty.dissipation + bty.bottom_dissipation + bty.vertical_pressure_work_sorted
    )
    print(f"loss terms: {loss_terms.data}")
    vel = loss_terms.data / -ty.FormDrag.data
    print(f"velocity: {vel}\n")

    print("without vertical pressure work term:")
    loss_terms = bty.dissipation + bty.bottom_dissipation
    print(f"loss terms: {loss_terms.data}")
    vel = loss_terms.data / -ty.FormDrag.data
    print(f"velocity: {vel}\n")


def _BottomDistance(ty):
    # depth of deepest v, t
    # -> generate z-matrix, set to nan where no v / t, then find largest z
    Z = np.tile(ty.z, (len(ty.dist), 1))
    Z = Z.T
    Zt = np.ma.masked_where(np.isnan(ty.t.values), Z)
    Zv = np.ma.masked_where(np.isnan(ty.v.values), Z)
    ztm = np.nanmax(Zt, axis=0)
    zvm = np.nanmax(Zv, axis=0)
    return ztm, zvm
