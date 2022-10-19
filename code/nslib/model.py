#!/usr/bin/env python
# coding: utf-8

"""
Various routines for analyzing the 2D MITgcm simulations of the
Samoan Passage Northern Sill overflow.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gvpy as gv
import pickle
from scipy import integrate
from tqdm import tqdm  # a little progress bar
import cmocean

import nslib as nsl


def load_initial_density():
    """Load initial model density field from model output.

    Returns
    -------
    rho : xr.DataArray
        Density at model start.
    """
    cfg = nsl.io.load_config()
    DataDir = cfg.model.input.full_model_run
    # read grid parameters and state variables
    b = xr.open_dataset(DataDir + "state.glob.nc")
    grid = xr.open_dataset(DataDir + "grid.glob.nc")
    b = xr.merge([b, grid])
    # extract temperature
    th = b.Temp.isel(T=0).squeeze().drop(["X", "T"])
    th = th.where(th.data != 0)  # set zeros to nan
    # add grid parameters
    th.coords["drF"] = grid.drF
    th.coords["dyF"] = grid.dyF.squeeze().drop(["X"])
    # calculate density
    talpha = 2e-4
    rho = rho0 * (1 - talpha * (th))
    return rho


def load():
    """Load model output into one `xarray.Dataset`.

    Returns
    -------
    b : xr.Dataset
        Dataset with model data.
    """
    cfg = nsl.io.load_config()
    DataDir = cfg.model.input.full_model_run
    grid = xr.open_dataset(DataDir + "grid.glob.nc")

    # read grid parameters, diagnostics output and state variables
    b6 = xr.open_dataset(DataDir + "diag1.glob.nc")
    b6s = xr.open_dataset(DataDir + "state.glob.nc")
    b62 = xr.open_dataset(DataDir + "diag2.glob.nc")

    # clean up and merge datasets
    b6 = b6.drop("UVEL")
    b6 = b6.squeeze("X")
    b6 = b6.drop(["Xp1", "X", "diag_levels", "KLviscAr", "WV_VEL", "momKE"])
    b6 = b6.rename({"Zmd000130": "Z", "Zld000130": "Zl"})
    b6["Z"] = (["Z"], grid.Z.values, grid.Z.attrs)
    b6["Zl"] = (["Zl"], grid.Zl.values, grid.Zl.attrs)
    b6 = b6.rename(
        {
            "VVEL": "VF",
            "WVEL": "WF",
            "THETA": "th",
            "KLeps": "eps",
            "KLdiffKr": "kappa",
            "PHIHYD": "phihyd",
        }
    )

    b62 = b62.drop("X")
    b62 = b62.squeeze()
    b62 = b62.drop("diag_levels")

    tmp = np.squeeze(b6s.phi_nh.values)
    if modelrun == "jesse":
        tmp = (tmp[1:-1, :, :] + tmp[:-2, :, :]) / 2
    else:
        tmp = (tmp[1:, :, :] + tmp[:-1, :, :]) / 2
    b6["phi_nh"] = (["T", "Z", "Y"], tmp)

    b = xr.merge([b6, b62, grid])
    b = b.rename({"ETAN": "eta", "PHIBOT": "phibot"})
    del [b6, b62, b6s, grid]

    # let's get rid of all stuff in x-direction
    b = b.squeeze()
    b = b.drop(
        ["X", "Xp1", "XG", "XC", "dxC", "dxF", "dxG", "dxV", "rAw", "HFacW"]
    )

    # New coords for Y and T
    b = b.assign_coords(dist=b.Y / 1000.0 - b.Y.mean() / 1000.0 + 13.0)
    b.dist.attrs["long_name"] = "Distance in km centered on ridge"
    b.dist.attrs["units"] = "km"
    b = b.assign_coords(time=b["T"] / 3600)
    b.time.attrs["long_name"] = "Time in hours"
    b.time.attrs["units"] = "h"

    # Load reference profile
    fileName = DataDir + "Tref"
    b["tref"] = (["Z"], np.fromfile(fileName))

    # dimensions
    m = b.Y.shape[0]
    n = b.Z.shape[0]
    tn = b["T"].shape[0]
    print("{0}x{1}x{2}".format(tn, n, m))

    # Bathy Mask
    BathyMaskNaN = np.squeeze(np.copy(b.HFacC.values))
    BathyMaskNaN[b.HFacC <= 0] = np.nan
    BathyMaskNaN[b.HFacC > 0] = 1
    b["BathyMask"] = (["Z", "Y"], BathyMaskNaN)

    # Calculate height of water column (depth + eta)
    b["H"] = b.eta + b.Depth

    b.attrs["modelrun"] = modelrun

    return b


def calculate_density(b):
    """Calculate model density.

    Parameters
    ----------
    b : xr.Dataset
        Model data

    Returns
    -------
    b : xr.Dataset
        Model data

    Notes
    -----
    Adds the following variables to `b`:

    - `rho` : model density
    - `rhop` : model density anomaly
    - `rhoref` : reference density profile
    """
    talpha = 2.0e-4
    tmp = b.th
    tmp = tmp.where(tmp.data != 0)  # set zeros to nan
    # This is how rho is defined in the MITgcm help pages, but the result is an
    # unstable water column:
    # b["rho"] = rho0 * (1 - talpha * (tmp - b.tref))
    # This is how rho looks reasonable. Not sure why, but it also makes sense
    # when defining density as rho = rho0 + rhoref + rhop
    b["rho"] = rho0 * (1 - talpha * tmp)
    b["rhop"] = -rho0 * talpha * (tmp - b.tref)
    b["rhoref"] = -rho0 * talpha * b.tref
    return b


def calculate_pressure(b):
    """Calculate model pressure.

    Calculates model pressure anomalies and absolute/in situ pressure.

    Parameters
    ----------
    b : xr.Dataset
        Model data

    Returns
    -------
    b : xr.Dataset
        Model data

    Notes
    -----
    Adds the following variables to `b`:

    - `PP` : pressure anomaly
    - `q` : non-hydrostatic pressure anomaly
    - `phibot` : bottom pressure anomaly
    - `p` : full pressure from integrated density and eta
    """
    # pressure perturbation as provided by model output
    b["PP"] = (b.phihyd * rho0).where(b.HFacC != 0)
    b.PP.attrs["description"] = "Pressure Anomaly"
    b.PP.attrs["units"] = "N/m^2"
    # non-hydrostatic pressure
    b["q"] = (b.phi_nh * rho0).where(b.HFacC != 0)
    b.q.attrs["description"] = "Non-hydrostatic Pressure Anomaly"
    b.q.attrs["units"] = "N/m^2"
    # convert bottom pressure potential anomaly into bottom pressure anomaly
    b["phibot"] = b.phibot * rho0
    b.phibot.attrs["description"] = "Bottom Pressure Anomaly"
    b.phibot.attrs["units"] = "N/m$^2$"
    # calculate full (hydrostatic) pressure by integrating density
    axis_Z = b.rho.get_axis_num("Z")
    surface_p = gBaro * rho0 * b.eta
    b["p"] = surface_p + np.cumsum(
        gravity * (b.rho * b.drF).where(b.HFacC != 0),
        axis=axis_Z,
    )
    b.p.attrs["description"] = "Pressure"
    b.p.attrs["units"] = "N/m$^2$"
    return b


def calculate_pressure_anomaly_sorted_rho(b, cfg):
    """Calculate pressure perturbation based on sorted initial stratification
    and reference profiles calculated thereof.

    Parameters
    ----------
    b : xarray Dataset
        Model grid and data
    cfg : box.Box
        Config read from `config.yml` via :class:`nslib.io.load_config`.
    """
    model_id = cfg.model.id
    file_sorted_profile = f"data/refrho_sorted_{model_id}.nc"
    ref_rho_sorted = xr.open_dataarray(file_sorted_profile)
    # Calculate reference pressure from the sorted density profile. The free
    # ocean surface $\eta=0$.
    ref_p = np.cumsum(gravity * (ref_rho_sorted * 20))
    # Interpolate reference density to model depths
    ref_rho_i = ref_rho_sorted.interp_like(b)
    rhop = b.rho - ref_rho_i
    # Calculate pressure perturbation from density anomaly.
    axis_Z = b.rho.get_axis_num("Z")
    pp = np.cumsum(gravity * (rhop * b.drF).where(b.Z < -1000), axis=axis_Z)
    pp = pp.drop("drF")
    b["PP_sorted_rho"] = pp
    return b


def calculate_velocities(b):
    """Calculate model velocities.

    Parameters
    ----------
    b : xr.Dataset
        Model data

    Returns
    -------
    b : xr.Dataset
        Model data

    Notes
    -----
    Adds the following variables to `b`:

    - `VC` : velocities at cell centers
    - `V` : barotropic horizontal velocity
    - `v` : baroclinic horizontal velocity
    - `WC` : vertical velocities at cell centers
    - `eps` : epsilon at cell centers
    - `W` : vertical velocity balancing horizontal flow
    - `wp` : residual vertical velocity (`WC`-`W`)
    """

    # interpolate to cell centers
    b["VC"] = (
        ["T", "Z", "Y"],
        (b.VF.values[:, :, 1:] + b.VF.values[:, :, :-1]) / 2,
    )
    # calculate barotropic horizontal velocity
    b["V"] = (
        ["T", "Y"],
        (
            (np.sum(b["VC"] * b.drF * b.HFacC, axis=1) + b.eta * b.VC.isel(Z=0))
            / b.H
        ).data,
    )
    # calculate a pseudo-barotropic velocity (time mean velocity field
    # calculated over the last 12 hours of the simulation)
    tmask = b.time > 60
    b["V2"] = (
        ["Z", "Y"],
        (b.VC.isel(T=tmask).mean(dim="T") * b.BathyMask).data,
    )
    # Baroclinic horizontal velocity
    b["v"] = (b["VC"] - b["V"]) * b.BathyMask
    b["v2"] = (b["VC"] - b["V2"]) * b.BathyMask

    # Vertical velocity
    WF = np.squeeze(b.WF)
    # interpolate from upper faces to cell centers. there are no velocities
    # defined at the bottom of the lowest cells. We'll simply repeat the
    # velocities from the cells above to end up with the right dimensions here.
    # np.dbstack() only works along the 3rd (last?) dimension, so we have to
    # roll axes...
    WFr = np.rollaxis(WF.values, 1, 3)
    WF2 = np.dstack((WFr, WF[:, -1, :]))
    WF2 = np.rollaxis(WF2, 2, 1)
    WC = (WF2[:, 1:, :] + WF2[:, :-1, :]) / 2
    b["WC"] = (["T", "Z", "Y"], WC)
    # Do the same for epsilon since it's also on that grid
    eps = np.squeeze(b.eps.values)
    epsr = np.rollaxis(eps, 1, 3)
    eps2 = np.dstack((epsr, eps[:, -1, :]))
    eps2 = np.rollaxis(eps2, 2, 1)
    b["eps"] = (["T", "Z", "Y"], (eps2[:, 1:, :] + eps2[:, :-1, :]) / 2)

    tmp = b.V * (b.Z + b.Depth) * b.BathyMask
    tmp = tmp.transpose("T", "Z", "Y")
    b["W"] = (["T", "Z", "Y"], -np.gradient(tmp, b.Y, axis=2))
    # residual vertical velocity
    b["wp"] = b.WC - b.W
    return b


def calculate_energetics(b):
    cfg = nsl.io.load_config()
    # full kinetic Energy
    Ek = 1 / 2 * rho0 * (b.VC ** 2 + b.WC ** 2) * b.BathyMask
    # barotropic horizontal kinetic energy
    Ehk0 = 0.5 * rho0 * b.V ** 2
    # bring to matrix form for plotting
    Ehk0m = Ehk0 * b.BathyMask
    Ehk0m = Ehk0m.transpose("T", "Z", "Y")
    # baroclinic kinetic Energy
    # Ekp = 0.5 * rho0 * (b.v ** 2 + b.WC ** 2) * b.BathyMask
    Ekp = 0.5 * rho0 * (b.v ** 2 + b.wp ** 2) * b.BathyMask
    # cross term
    Ehk0p = 0.5 * rho0 * b.v * b.V * b.BathyMask
    # perturbation potential energy
    Ep0 = 0.5 * rho0 * gBaro * b.eta ** 2
    # combine energy terms into one dataset
    E = xr.Dataset(
        {"Ek": Ek, "Ehk0": Ehk0, "Ekp": Ekp, "Ehk0p": Ehk0p, "Ep0": Ep0}
    )

    # available potential energy
    if cfg.model.output.ape.exists() is False:
        LAPE, ZETA = LambAPEnew(b)
        lape = xr.DataArray(LAPE)
        lape.to_netcdf(cfg.model.output.ape)
        zeta = xr.DataArray(ZETA)
        zeta.to_netcdf(cfg.model.output.ape_zeta)
    else:
        LAPE = xr.open_dataarray(cfg.model.output.ape).data
        ZETA =  xr.open_dataarray(cfg.model.output.ape_zeta).data
    E["Epp"] = (["T", "Z", "Y"], LAPE)

    # Available potential energy calculated against a density profile based on
    # a sorted density field
    if cfg.model.output.ape_sorted.exists() is False:
        LAPE_sorted_rho, ZETA_sorted_rho = LambAPEnew(b, sorted_ref_rho=True)
        lapes = xr.DataArray(LAPE_sorted_rho)
        lapes.to_netcdf(cfg.model.output.ape_sorted)
    else:
        LAPE_sorted_rho = xr.open_dataarray(cfg.model.output.ape_sorted).data
    E["Epp_sorted_rho"] = (["T", "Z", "Y"], LAPE_sorted_rho)

    # Buoyancy Flux and BT/BC Conversion
    E["BuoyancyFlux"] = gravity * b.rhop * b.WC
    E["Conversion"] = gravity * b.rhop * b.W
    E["nhConversion"] = np.gradient(b.q, b.Z, axis=1) * b.W

    # Internal Wave Fluxes
    E["IWEnergyFluxHoriz"] = b.v * (b.PP)
    E["nhIWEnergyFluxHoriz"] = b.v * b.q
    E["IWEnergyFluxVert"] = b.wp * b.PP
    E["nhIWEnergyFluxVert"] = b.wp * b.q

    # Calculate the horizontal fluxes based on a reference density and pressure
    # profile that is based on a sorted field of initial stratification.
    E["IWEnergyFluxHoriz_sorted_rho"] = b.v * b.PP_sorted_rho
    # And also vertical flux
    E["IWEnergyFluxVert_sorted_rho"] = b.wp * b.PP_sorted_rho

    # Bottom drag
    # find bottom cells
    tmp = b.HFacC.values
    BottomCells = np.zeros_like(b.Y.values, dtype="int")
    for i, column in enumerate(tmp.T):
        indices = np.argwhere(column > 0)
        BottomCells[i] = indices[-1]
    BottomDrag = np.zeros_like(b.eta) * np.nan
    for j, d in enumerate(BottomCells):
        # not sure how the Kang formulation ensures that this is always
        # positive... putting abs around it...
        BottomDrag[:, j] = np.abs(
            rho0
            * Cd
            * abs(b.VC.values[:, d, j])
            * (
                b.v.values[:, d, j] * b.VC.values[:, d, j]
                + b.WC.values[:, d, j] ** 2
            )
        )
    E["BottomDrag"] = (["T", "Y"], BottomDrag)

    # Bottom drag with velocities 40m above the bottom
    above = -b.Depth + 40
    tmp1 = b.v.where(b.Z>above, other=-1e6)
    tmp = tmp1.isel(T=0)
    BottomCellsAbove = np.zeros_like(b.Y.values, dtype="int")
    for i, column in enumerate(tmp.T):
        indices = np.argwhere(column.data > -1e6)
        BottomCellsAbove[i] = indices[-1]
    BottomDragAbove = np.zeros_like(b.eta) * np.nan
    for j, d in enumerate(BottomCellsAbove):
        BottomDragAbove[:, j] = np.abs(
            rho0
            * Cd
            * abs(b.VC.values[:, d, j])
            * (
                b.v.values[:, d, j] * b.VC.values[:, d, j]
                + b.WC.values[:, d, j] ** 2
            )
        )
    E["BottomDragAbove"] = (["T", "Y"], BottomDragAbove)

    # diffusive flux of kinetic energy
    # -grad(nu(grad(Ek))
    # units are 1/m^2 m^2/s J/m = W/m
    E["diffEkVert"] = (
        ["T", "Z", "Y"],
        np.gradient(b.kappa * np.gradient(E.Ekp, b.Z, axis=1), b.Z, axis=1),
    )

    # diffusive flux of potential energy
    # -grad(kappa g zeta grad(rhop))
    E["diffEpVert"] = (
        ["T", "Z", "Y"],
        -1
        * np.gradient(
            b.kappa * gravity * ZETA * np.gradient(b.rhop, b.Z, axis=1),
            b.Z,
            axis=1,
        ),
    )

    return E


def calculate_bernoulli_energetics(cfg, b, zlim=0):
    """Calculate energy terms of a simple Bernoulli energy equation.

    The terms are integrated over the full depth and their divergence is simply
    calculated as the difference between left and right side of the control
    volume. They should be balanced by diffusive terms.
    """
    Ek = 1 / 2 * rho0 * (b.VC ** 2 + b.WC ** 2) * b.BathyMask
    Ep = b.rhop * gravity * b.Z * b.BathyMask
    p = b.p * b.BathyMask
    # horizontal integration limits
    y1 = cfg.parameters.model.integration_horizontal_min
    y2 = cfg.parameters.model.integration_horizontal_max
    print(f"horizontal integration limits {y1} and {y2}")
    # mask for horizontal limits
    ym = (b.dist > y1) & (b.dist < y2)
    # mask for vertical limit. z is negative. find all z smaller than zlim.
    if zlim > 0:
        zlim = -1 * zlim
    zm = b.Z < zlim
    # pre-allocate B as xarray Dataset by copying some grid info
    B = xr.Dataset(
        {"dist": b.dist, "Y": b.Y, "Z": b.Z, "T": b["T"], "time": b["time"]}
    )

    # Unsteadiness - first volume integral (gives J/m)
    EkpVI = (Ek.isel(Y=ym, Z=zm) * b.dyF.isel(Y=ym) * b.drF.isel(Z=zm)).sum(
        dim=("Z", "Y")
    )
    EppVI = (Ep.isel(Y=ym, Z=zm) * b.dyF.isel(Y=ym) * b.drF.isel(Z=zm)).sum(
        dim=("Z", "Y")
    )
    # Unsteadiness -  time derivative
    B["dEkpdt"] = (["T"], np.gradient(EkpVI, (b.time * 3600)))
    B.dEkpdt.attrs["name"] = "Unsteadiness KE"
    B.dEkpdt.attrs["units"] = "W/m"
    B["dEppdt"] = (["T"], np.gradient(EppVI, (b.time * 3600)))
    B.dEppdt.attrs["name"] = "Unsteadiness APE"
    B.dEppdt.attrs["units"] = "W/m"

    # Energy advection - horizontal
    EkpFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * Ek.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    EppFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * Ep.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    EpFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * p.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    B["EkpFluxDivergence"] = EkpFluxDI.isel(Y=-1) - EkpFluxDI.isel(Y=0)
    B.EkpFluxDivergence.attrs["name"] = "EK Flux Divergence"
    B.EkpFluxDivergence.attrs["units"] = "W/m"
    B["EppFluxDivergence"] = EppFluxDI.isel(Y=-1) - EppFluxDI.isel(Y=0)
    B.EppFluxDivergence.attrs["name"] = "APE Flux Divergence"
    B.EppFluxDivergence.attrs["units"] = "W/m"
    B["EpFluxDivergence"] = EpFluxDI.isel(Y=-1) - EpFluxDI.isel(Y=0)
    B.EppFluxDivergence.attrs["name"] = "Static Pressure Energy Flux Divergence"
    B.EppFluxDivergence.attrs["units"] = "W/m"

    # Dissipation
    B["Dissipation"] = (
        rho0
        * b.eps.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.dyF.isel(Y=ym)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim=("Z", "Y"))
    B.Dissipation.attrs["name"] = "Integrated Dissipation"
    B.Dissipation.attrs["units"] = "W/m"

    return Ek, Ep, p, B


def sorted_background_density(rerun_sort=False, plot=False):
    """Sort initial density into stable background state and return density profile.

    Parameters
    ----------
    rerun_sort : bool, optional
        Rerun density sorting. Defaults to False and loads saved data instead.
    plot : bool, optional
        Plot result. Defaults to False.

    Returns
    -------
    ref_rho_sorted : xr.DataArray
        Mean density profile from sorted background state.
    """
    cfg = nsl.io.load_config()
    file_sorted_density_field = cfg.model.output.sorted_initial_density
    rho = load_initial_density()
    # Interpolate the density field to a regular grid with 20m resolution in
    # the horizontal and vertical.
    y0 = rho.Y[0] - rho.dyF[0] / 2
    y1 = rho.Y[-1] + rho.dyF[-1] / 2
    ynew = np.arange(y0 + 10, y1, 20)
    # note: Z is negativ in the ocean
    z0 = rho.Z[0] + rho.drF[0] / 2
    z1 = rho.Z[-1] - rho.drF[-1] / 2
    znew = np.arange(z0 - 10, z1, -20)
    # generate dummy DataArray for interpolation
    dummy = np.full((len(znew), len(ynew)), np.nan)
    tmp = xr.DataArray(dummy, dims=["Z", "Y"], coords={"Z": znew, "Y": ynew})
    r = rho.interp_like(tmp)
    # A few profiles near the upstream and downstream edges are all nan since
    # they are outside the interpolation range. Fill them.
    ii = np.isnan(r[50, :]).data
    ii = np.flatnonzero(ii)
    ileft = ii[ii < 100]
    for i in ileft:
        r[:, i] = r[:, ileft[-1] + 1]
    iright = ii[ii > 1000]
    for i in iright:
        r[:, i] = r[:, iright[0] - 1]
    # Also need to adjust top and bottom. Top is easy, but bottom may be hard
    # to adjust for topography... However, the bottom shouldn't be missing
    # anything since we have 20m resolution there! Only need to fill the top.
    ii = np.isnan(r[:, 100]).data
    ii = np.flatnonzero(ii)
    itop = ii[ii < 100]
    for i in itop:
        r[i, :] = r[itop[-1] + 1, :]
    if rerun_sort:
        # Reshape density into 1D vector
        rv = np.squeeze(np.reshape(r.data, (1, -1)))
        # Remove bottom values
        rv = rv[~np.isnan(rv)]
        # Sort density vector
        rvs = np.sort(rv, kind="stable")
        # Convert to list to use pop()
        rvsl = rvs.tolist()
        # Now fill the matrix with the sorted densities, starting with the densest
        # at the bottom and upstream
        rsorted = r.copy()
        rshape = rsorted.shape
        zi = range(rshape[0] - 1, 0, -1)
        yj = range(rshape[1])
        for i in tqdm(zi):
            for j in yj:
                if ~np.isnan(r[i, j]):
                    rsorted[i, j] = rvsl.pop()
        # Save sorted field
        rsorted.to_netcdf(file_sorted_density_field)
    else:
        rsorted = xr.open_dataarray(file_sorted_density_field)

    # Plot upstream / downstream / sorted
    if plot:
        fig, ax = gv.plot.quickfig()
        rsorted.isel(Y=100).plot(y="Z")
        r.isel(Y=2).plot(y="Z")
        rsorted.isel(Y=-100).plot(y="Z")
        r.isel(Y=-2).plot(y="Z")
        rsorted.mean(dim="Y").plot(color="k", y="Z")
        ax.set(ylim=(-5300, -3500), xlim=(999.55, 999.68))

    # Interpolate back to original model resolution
    rsortedi = rsorted.interp_like(r)
    # Calculate mean profile from the sorted field
    ref_rho_sorted = rsortedi.mean(dim="Y")
    # Save mean profile
    file_sorted_profile = cfg.model.output.refrho_sorted
    ref_rho_sorted.attrs["title"] = "refrho_sorted"
    ref_rho_sorted.to_netcdf(file_sorted_profile)

    return ref_rho_sorted


def LambAPE(b):
    refrho = b.rho.isel(T=0, Y=-1).data
    # need to reread the model data if not analyzing from the model
    # initialization but a later point in time.
    if b.modelrun == "jesse100150":
        read_rhoref = False
        if read_rhoref:
            # need to read model at beginning. do this only once and save the
            # reference profile.
            print("reading reference density profile at model start time")
            model_id = "jesse"
            tmp = load(model_id)
            trange = range(5)
            tmp = tmp.isel(T=trange)
            tmp = calculate_density(tmp)
            refrho = tmp.rho.isel(T=0, Y=-1)
            refrho.to_netcdf("data/refrho.nc")
        else:
            refrho = xr.open_dataarray("data/refrho.nc")
        # interpolate to model dataset in case we are using a subset
        refrho = refrho.interp_like(b).data
    refz = b.Z.data
    LAPE = b.rho.data * np.nan
    ZETA = b.rho.data * np.nan
    # loop over all time steps
    for i, ti in tqdm(enumerate(b["T"])):
        # loop over all columns in density matrix
        for j, rho in enumerate(b.rho.isel(T=i).data.T):
            for k, (r, z) in enumerate(zip(rho, refz)):
                if ~np.isnan(r):
                    # find index of density in reference density profile
                    i2 = gv.misc.nearidx(refrho, r)
                    # find index of reference density
                    iz = gv.misc.nearidx(np.abs(refz), np.abs(z))
                    if iz < i2:
                        H = -1 * (refz[iz:i2] - z)
                        drho = r - refrho[iz:i2]
                    else:
                        H = -1 * (refz[i2:iz] - z)
                        drho = r - refrho[i2:iz]
                    LAPE[i, k, j] = gravity * integrate.trapz(drho, H)
                    # zeta is depth in perturbed state minus depth in reference state
                    ZETA[i, k, j] = refz[iz] - refz[i2]
    return LAPE, ZETA


def LambAPEnew(b, sorted_ref_rho=False):
    cfg = nsl.io.load_config()
    # read reference density from model initialization
    if sorted_ref_rho:
        ref_rho = xr.open_dataarray(cfg.model.output.refrho_sorted)
    else:
        ref_rho = xr.open_dataarray(cfg.model.output.refrho)
    # interpolate to model dataset in case we are using a subset
    ref_rho = ref_rho.interp_like(b).data
    # TODO: make sure deepest refrho depth level covers deepest rho.
    # maximum density in the reference density profile
    ref_rho_max = np.max(ref_rho).data
    # index of deepest reference density
    ref_rho_n = len(ref_rho) - 1
    z = b.Z.data
    z_max = b.Z.max().data
    drF = b.drF.data
    ape = b.rho.data * np.nan
    zeta = b.rho.data * np.nan
    # loop over all time steps
    for i, ti in tqdm(enumerate(b["T"])):
        # loop over all columns in density matrix
        for j, rho in enumerate(b.rho.isel(T=i).data.T):
            # loop over each depth level / fluid particle in the water column
            for k, (rho_k, z_k) in enumerate(zip(rho, z)):
                ape[i, k, j], zeta[i, k, j] = ape_fluid_particle(
                    rho_k, k, ref_rho, z, drF, rho_z_is_index=True, plot=False
                )
    return ape, zeta


def ape_fluid_particle(
    rho, rho_z, ref_rho_profile, ref_rho_z, drF, rho_z_is_index=True, plot=False
):
    """Calculate APE for one fluid particle relative to a reference profile.

    Parameters
    ----------
    rho : float
        Density of fluid particle.
    rho_z : int or float
        Fluid particle depth. Depending on rho_z_is_index this is either and
        index into ref_rho_z or directly the particle depth. The former is
        faster in computation.
    ref_rho_profile : array-like
        Reference density profile.
    ref_rho_z : array-like
        Reference density profile depth vector.
    drF : array-like
        Vertical grid cell size.
    rho_z_is_index : bool, optional
        Indicates whether rho_z is an index into ref_rho_z, defaults to True.

    Returns
    -------
    ape : float
        APE of fluid particle
    zeta : float
        Fluid particle displacement relative to reference state.
    """
    if np.isnan(rho):
        return np.nan, np.nan

    # depth of fluid particle as index into ref_rho_z
    k = rho_z if rho_z_is_index else gv.misc.nearidx(ref_rho_z, rho_z)
    rho_z = ref_rho_z[k] if rho_z_is_index else rho_z

    ref_rho_max = np.max(ref_rho_profile)
    # index of deepest reference density
    ref_rho_n = len(ref_rho_profile) - 1
    # find index of density in reference density profile
    if rho >= ref_rho_max:
        # density greater than max reference density; select
        # deepest index
        # k_star is depth index of reference state;
        # k depth index of perturbed state.
        k_star = ref_rho_n
        excess_rho = rho - ref_rho_max
        reached_end = True
    else:
        # find index in reference density profile
        k_star = gv.misc.nearidx(ref_rho_profile, rho)
        reached_end = False
    # integrate from zstar to z over density difference between
    # perturbed and reference state
    if k < k_star:
        # H = -1 * (z[k:k_star] - z_k)
        H = np.cumsum(drF[k:k_star])
        diff_rho = rho - ref_rho_profile[k:k_star]
    else:
        # H = -1 * (z[k_star:k] - z_k)
        H = np.cumsum(drF[k_star:k])
        diff_rho = rho - ref_rho_profile[k_star:k]
    ape = gravity * integrate.trapz(diff_rho, H)
    # zeta is depth in reference state minus depth in perturbed state
    zeta = rho_z - ref_rho_z[k_star]

    if plot:
        fig, ax = gv.plot.quickfig()
        ax.plot(ref_rho_profile, ref_rho_z, color="0.5", linewidth=0.8)
        ax.vlines(rho, rho_z, ref_rho_z[k_star], color="orange", linestyle="--")
        ax.hlines(
            rho_z, ref_rho_profile[k], rho, color="orange", linestyle="--"
        )
        ax.plot(rho, rho_z, marker="o", color="orange")
        ax.text(
            0.1,
            0.1,
            f"rhodiff={rho - ref_rho_profile[k]:1.5f}kg/m^3",
            transform=ax.transAxes,
        )
        ax.text(0.1, 0.2, f"zeta={zeta:1.0f}m", transform=ax.transAxes)

    return ape, zeta


def bernoulli_function(b, th_interface=0.9, quick_d=True):
    """Calculate Bernoulli function transport along the flow.

    Parameters
    ----------
    b : xr.Dataset
        Model data

    Returns
    -------
    TODO

    """
    # Select only the deeper part of the water column
    # Switch to distance coordinate
    c = b.where(b.Z < -3000, drop=True).swap_dims({"Y": "dist"})
    # Density anomaly
    # For now we can average density below 4600m and above 4200m depth.
    rho = c.rho * c.BathyMask
    rho_l = rho.where(rho.Z < -4600).mean(dim="Z")
    rho_u = rho.where((rho.Z > -4200) * (rho.Z < -3700)).mean(dim="Z")
    drho = rho_l - rho_u
    # Bottom elevation h.
    depth = c.Depth.isel(Z=0)
    h = np.abs((depth - depth.min()) - (depth - depth.min()).max())
    # Interface definition.
    mask = c.th < th_interface
    if quick_d:
        # Determine layer thickness quickly.
        thickness = (c.HFacC * c.drF).where(mask).sum(dim='Z')
    else:
        # find depth of interface and subtract from bottom depth for layer thickness
        z = c.Z
        interface_z = []
        for dd, x in c.th.groupby('dist'):
        #     xi = ~np.isnan(x)
            interface_z.append(sp.interpolate.interp1d(x, z, axis=0, bounds_error=False)(model_interface))

        thickness = np.array(interface_z) + depth
    d = thickness
    # Layer mean velocity.
    v = ((c.HFacC * c.VC)*(c.HFacC*c.drF)).where(mask).sum(dim='Z')/d
    # Bernoulli function along the flow
    gprime = drho / 999 * 9.81
    B = v ** 3 * d / 2 + v * gprime * (d ** 2 + h * d)
    # Convert to units of kW/m
    B = rho0 * B / 1000

    return B


# energy budgets
def energy_budget_box(cfg, b, E, zlim=0):
    """Calculate various terms for the baroclinic energy equation integrated
    either over a box of variable width and full water column or a box with
    variable width and height.

    Parameters
    ----------
    cfg : box.Box
        Config read from `config.yml` via :class:`nslib.io.load_config`.
    b : xarray Dataset
        Model grid and data
    E : xarray Dataset
        Energy fields calculated from b.
    zlim : float, optional
        Depth determining the height of the box if provided.
        Otherwise the box spans the full water column.

    Returns
    -------
    B : xarray Dataset
        Terms of the baroclinic energy equation

    Notes
    -----
    Horizontal integration limits are read from `config.yml`.

    Internal wave fluxes come in several flavors depending on how primed
    quantities are calculated:

    1. Based on a downstream reference density profile, following Kang (2010) {cite}`kang10`.
    2. Based on locally calculated reference density, velocity profiles.
    3. Based on high-pass filtered time series of pressure, velocity.
    """
    y1 = cfg.parameters.model.integration_horizontal_min
    y2 = cfg.parameters.model.integration_horizontal_max
    print(f"horizontal integration limits {y1} and {y2}")

    # mask for horizontal limits
    ym = (b.dist > y1) & (b.dist < y2)

    # mask for vertical limit. z is negative. find all z smaller than zlim.
    if zlim > 0:
        zlim = -1 * zlim
    zm = b.Z < zlim

    # pre-allocate B as xarray Dataset by copying some grid info
    B = xr.Dataset(
        {"dist": b.dist, "Y": b.Y, "Z": b.Z, "T": b["T"], "time": b["time"]}
    )

    # Unsteadiness - first volume integral (gives J/m)
    EkpVI = (E.Ekp.isel(Y=ym, Z=zm) * b.dyF.isel(Y=ym) * b.drF.isel(Z=zm)).sum(
        dim=("Z", "Y")
    )
    EppVI = (E.Epp.isel(Y=ym, Z=zm) * b.dyF.isel(Y=ym) * b.drF.isel(Z=zm)).sum(
        dim=("Z", "Y")
    )
    # Unsteadiness -  time derivative
    B["dEkpdt"] = (["T"], np.gradient(EkpVI, (b.time * 3600)))
    B.dEkpdt.attrs["name"] = "Unsteadiness KE"
    B.dEkpdt.attrs["units"] = "W/m"
    B["dEppdt"] = (["T"], np.gradient(EppVI, (b.time * 3600)))
    B.dEppdt.attrs["name"] = "Unsteadiness APE"
    B.dEppdt.attrs["units"] = "W/m"
    # Unsteadiness of perturbation potential energy
    axisT = b.eta.get_axis_num(dim="T")
    B["dEp0dt"] = (
        rho0
        * gBaro
        * b.eta.isel(Y=ym)
        * np.gradient(b.eta.isel(Y=ym), b.time * 3600, axis=axisT)
        * b.dyF.isel(Y=ym)
    ).sum(dim="Y")
    # combine unsteadiness terms
    B["Unsteadiness"] = B.dEkpdt + B.dEppdt + B.dEp0dt

    # BT/BC Conversion
    B["Conversion"] = (
        E.Conversion.isel(Y=ym, Z=zm) * b.dyF.isel(Y=ym) * b.drF.isel(Z=zm)
    ).sum(dim=("Z", "Y"))
    B.Conversion.attrs["name"] = "BT/BC Conversion"
    B.Conversion.attrs["units"] = "W/m"

    # Energy advection - horizontal
    EkpFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * E.Ekp.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    EppFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * E.Epp.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    EppSortedFluxDI = (
        b.v.isel(Y=ym, Z=zm)
        * E.Epp_sorted_rho.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim="Z")
    B["EkpFluxDivergence"] = EkpFluxDI.isel(Y=-1) - EkpFluxDI.isel(Y=0)
    B.EkpFluxDivergence.attrs["name"] = "EK Flux Divergence"
    B.EkpFluxDivergence.attrs["units"] = "W/m"
    B["EppFluxDivergence"] = EppFluxDI.isel(Y=-1) - EppFluxDI.isel(Y=0)
    B.EppFluxDivergence.attrs["name"] = "APE Flux Divergence"
    B.EppFluxDivergence.attrs["units"] = "W/m"
    B["EppSortedFluxDivergence"] = EppSortedFluxDI.isel(
        Y=-1
    ) - EppSortedFluxDI.isel(Y=0)
    B.EppSortedFluxDivergence.attrs["name"] = "APE Flux Divergence"
    B.EppSortedFluxDivergence.attrs["units"] = "W/m"
    # Energy advection - vertical
    B["EkpVFluxDivergence"] = (
        b.wp.isel(Z=np.argmax(zm.values), Y=ym)
        * E.Ekp.isel(Z=np.argmax(zm.values), Y=ym)
        * b.dyF.isel(Y=ym)
    ).sum(dim="Y")
    B["EppVFluxDivergence"] = (
        b.wp.isel(Z=np.argmax(zm.values), Y=ym)
        * E.Epp.isel(Z=np.argmax(zm.values), Y=ym)
        * b.dyF.isel(Y=ym)
    ).sum(dim="Y")

    # Internal wave fluxes - horizontal
    # hydrostatic
    IWFluxHorizDI = (
        E.IWEnergyFluxHoriz.isel(Y=ym, Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
        * b.drF.isel(Z=zm)
    ).sum(dim="Z")
    B["IWFluxHorizDivergence"] = IWFluxHorizDI.isel(Y=-1) - IWFluxHorizDI.isel(
        Y=0
    )
    B.IWFluxHorizDivergence.attrs["name"] = (
        "Horizontal Internal Wave " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence.attrs["units"] = "W/m"
    # horizontal flux based on sorted density field
    IWFluxHorizSortedDI = (
        E.IWEnergyFluxHoriz_sorted_rho.isel(Y=ym, Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
        * b.drF.isel(Z=zm)
    ).sum(dim="Z")
    B["IWSortedFluxHorizDivergence"] = IWFluxHorizSortedDI.isel(
        Y=-1
    ) - IWFluxHorizSortedDI.isel(Y=0)
    B.IWSortedFluxHorizDivergence.attrs["name"] = (
        "Horizontal Internal Wave " + "Flux Divergence"
    )
    B.IWSortedFluxHorizDivergence.attrs["units"] = "W/m"
    # non-hydrostatic
    IWFluxHorizDInh = (
        E.nhIWEnergyFluxHoriz.isel(Y=ym, Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
        * b.drF.isel(Z=zm)
    ).sum(dim="Z")
    B["IWFluxHorizDivergencenh"] = IWFluxHorizDInh.isel(
        Y=-1
    ) - IWFluxHorizDInh.isel(Y=0)
    B.IWFluxHorizDivergencenh.attrs["name"] = (
        "non-hydrostatic Horizontal " + "Internal Wave Flux Divergence"
    )
    B.IWFluxHorizDivergencenh.attrs["units"] = "W/m"
    # based on local mean profiles
    IWFluxHorizDI_lp = (
        E.lp_hwf.isel(Y=ym, Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
        * b.drF.isel(Z=zm)
    ).sum(dim="Z")
    B["IWFluxHorizDivergence_lp"] = IWFluxHorizDI_lp.isel(
        Y=-1
    ) - IWFluxHorizDI_lp.isel(Y=0)
    B.IWFluxHorizDivergence_lp.attrs["name"] = (
        "Horizontal Internal Wave Local Profile " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence.attrs["units"] = "W/m"
    # based on high-pass filtered time series
    IWFluxHorizDI_hp = (
        E.hp_hwf.isel(Y=ym, Z=zm)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
        * b.drF.isel(Z=zm)
    ).sum(dim="Z")
    B["IWFluxHorizDivergence_hp"] = IWFluxHorizDI_hp.isel(
        Y=-1
    ) - IWFluxHorizDI_hp.isel(Y=0)
    B.IWFluxHorizDivergence_hp.attrs["name"] = (
        "Horizontal Internal Wave High Pass Filter " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence.attrs["units"] = "W/m"

    # Internal wave & diffusive fluxes - vertical
    vars = [
        "IWFluxVertDivergence",
        "IWSortedFluxVertDivergence",
        "IWFluxVertDivergencenh",
        "IWFluxVertDivergence_lp",
        "IWFluxVertDivergence_hp",
        "diffEkFluxVertDivergence",
        "diffEpFluxVertDivergence",
    ]
    if zlim != 0:
        B["IWFluxVertDivergence"] = (
            E.IWEnergyFluxVert.isel(Z=np.argmax(zm.values), Y=ym)
            * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["IWSortedFluxVertDivergence"] = (
            E.IWEnergyFluxVert_sorted_rho.isel(Z=np.argmax(zm.values), Y=ym)
            * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["IWFluxVertDivergencenh"] = (
            E.nhIWEnergyFluxVert.isel(Z=np.argmax(zm.values), Y=ym)
            * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["IWFluxVertDivergence_lp"] = (
            E.lp_vwf.isel(Z=np.argmax(zm.values), Y=ym) * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["IWFluxVertDivergence_hp"] = (
            E.hp_vwf.isel(Z=np.argmax(zm.values), Y=ym) * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["diffEkFluxVertDivergence"] = (
            E.diffEkVert.isel(Z=np.argmax(zm.values), Y=ym) * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
        B["diffEpFluxVertDivergence"] = (
            E.diffEpVert.isel(Z=np.argmax(zm.values), Y=ym) * b.dyF.isel(Y=ym)
        ).sum(dim="Y")
    else:
        # no vertical flux divergence if full water column
        for var in vars:
            B[var] = (
                ["T"],
                np.zeros_like(B.IWFluxHorizDivergence),
            )
    B.IWFluxVertDivergence.attrs["name"] = (
        "Vertical Internal Wave " + "Flux Divergence"
    )
    B.IWSortedFluxVertDivergence.attrs["name"] = (
        "Vertical Internal Wave (sorted)" + "Flux Divergence"
    )
    B.IWFluxVertDivergencenh.attrs["name"] = (
        "non-hydrostatic Vertical " + "Internal Wave Flux Divergence"
    )
    B.IWFluxVertDivergence_lp.attrs["name"] = (
        "Vertical Internal Wave Local Profile " + "Flux Divergence"
    )
    B.IWFluxVertDivergence_hp.attrs["name"] = (
        "Vertical Internal Wave High Pass Filter " + "Flux Divergence"
    )
    for var in vars:
        B[var].attrs["units"] = "W/m"

    # Dissipation
    B["Dissipation"] = (
        rho0
        * b.eps.isel(Y=ym, Z=zm)
        * b.drF.isel(Z=zm)
        * b.dyF.isel(Y=ym)
        * b.HFacC.isel(Y=ym).isel(Z=zm)
    ).sum(dim=("Z", "Y"))
    B.Dissipation.attrs["name"] = "Integrated Dissipation"
    B.Dissipation.attrs["units"] = "W/m"

    # Bottom Drag
    B["BottomDrag"] = (E.BottomDrag.isel(Y=ym) * b.dyF.isel(Y=ym)).sum(dim="Y")
    B.BottomDrag.attrs["name"] = "Integrated Bottom Drag"
    B.BottomDrag.attrs["units"] = "W/m"

    print("Calculating bottom drag [from vels 40m above bottom] term...")
    B["BottomDragAbove"] = (E.BottomDragAbove.isel(Y=ym) * b.dyF.isel(Y=ym)).sum(dim="Y")
    B.BottomDragAbove.attrs["name"] = "Integrated Bottom Drag"
    B.BottomDragAbove.attrs["units"] = "W/m"

    # Combine a few of the terms
    B["IWFDH"] = B.IWFluxHorizDivergence_hp + B.IWFluxHorizDivergencenh
    B["IWFDV"] = B.IWFluxVertDivergence_hp + B.IWFluxVertDivergencenh
    # B["IWFDH"] = B.IWFluxHorizDivergence_hp
    # B["IWFDV"] = B.IWFluxVertDivergence_hp
    B["RHS"] = (
        B.IWFDH
        + B.IWFDV
        + B.Unsteadiness
        + B.EkpFluxDivergence
        + B.EkpVFluxDivergence
        + B.EppFluxDivergence
        + B.EppVFluxDivergence
        + B.Dissipation
        + B.BottomDrag
    )

    B.attrs["description"] = (
        "Energy budget within a box defined by\n"
        + "up/downstream distance and depth level"
    )
    B.attrs["ylims"] = [y1, y2]
    B.attrs["zlim"] = zlim

    return B


def energy_budget_layer(
    cfg, b, E, isot, avg_up_dn=False, steadiness_terms_only=False
):
    r"""Calculate various terms for the baroclinic energy equation integrated
    over a layer.

    Parameters
    ----------
    cfg : box.Box
        Config read from `config.yml` via :class:`nslib.io.load_config`.
    b : xarray Dataset
        Model grid and data
    E : xarray Dataset
        Energy fields calculated from b.
    isot : float, optional
        Temperature determining the upper isotherm to be used as boundary.
    avg_up_dn : bool, optional
        Average over 2km upstream and downstream when calculating horizontal
        divergence of potential and kinetic energy flux. Defaults to False.
    steadiness_terms_only : bool, optional
        Set to true to only output steadiness terms. Defaults to False.

    Returns
    -------
    B : xarray Dataset
        Terms of the baroclinic energy equation

    Notes
    -----
    Horizontal integration limits are read from `config.yml`.

    Internal wave fluxes come in several flavors depending on how primed
    quantities are calculated:

    1. Based on a downstream reference density profile, following Kang (2010) {cite}`kang10`.
    2. Based on locally calculated reference density, velocity profiles.
    3. Based on high-pass filtered time series of pressure, velocity.
    """
    # Read horizontal integration limits
    y1 = cfg.parameters.model.integration_horizontal_min
    y2 = cfg.parameters.model.integration_horizontal_max
    print(f"horizontal integration limits {y1} and {y2}")

    # mask for horizontal limits
    ym = (b.dist > y1) & (b.dist < y2)

    # add temperature to E for masking
    E["th"] = b.th

    # add epsilon, VC, WC, wp, eta to E
    E["eps"] = b.eps
    E["VC"] = b.VC
    E["WC"] = b.WC
    E["wp"] = b.wp
    # E["eta"] = (['T', 'Y'], b.eta)

    # drop variables we don't need here to speed up the whole thing a bit
    drop_vars = ["Ek", "Ehk0", "Ehk0p", "nhConversion"]
    for dv in drop_vars:
        try:
            E = E.drop(dv)
        except:
            pass

    # reduce E to region within ylims,
    # blank out everything warmer than isot
    print("Reducing dataset to selected region...")
    Er = E.where(E.th < isot).isel(Y=ym)

    # masks for upstream / downstream averaging when calculating the divergence terms
    yup = (Er.dist > y1) & (Er.dist < y1 + 2)
    ydn = (Er.dist > y2 - 2) & (Er.dist < y2)

    # need to treat BottomDrag and eta differently since they don't have Z-dimension
    Er["BottomDrag"] = E.BottomDrag.isel(Y=ym)
    Er["BottomDragAbove"] = E.BottomDragAbove.isel(Y=ym)
    # Er["eta"] = (["T", "Y"], b.eta.isel(Y=ym))
    Er["eta"] = b.eta.isel(Y=ym)

    # reduce dyF
    dyF = b.dyF.isel(Y=ym)
    drF = b.drF
    HFacC = b.HFacC.isel(Y=ym)

    # pre-allocate B as xarray Dataset by copying some grid info
    B = xr.Dataset(
        {"dist": b.dist, "Y": b.Y, "Z": b.Z, "T": b["T"], "time": b["time"]}
    )

    # find indices of the interface. we'll use this for the vertical fluxes.
    interface = (
        E.th.isel(Y=ym).where(E.th.isel(Y=ym) < isot, 99) != 99
    ).argmax(dim="Z")

    print("Calculating unsteadiness term...")
    # Unsteadiness - first volume integral (gives J/m)
    EkpVI = (Er.Ekp * dyF * drF).sum(dim=("Z", "Y"))
    EppVI = (Er.Epp * dyF * drF).sum(dim=("Z", "Y"))
    # Unsteadiness -  time derivative - [W/m]
    B["dEkpdt"] = (["T"], np.gradient(EkpVI, (b.time * 3600)))
    B.dEkpdt.attrs["name"] = "Unsteadiness KE"
    B.dEkpdt.attrs["units"] = "W/m"
    B["dEppdt"] = (["T"], np.gradient(EppVI, (b.time * 3600)))
    B.dEppdt.attrs["name"] = "Unsteadiness APE"
    B.dEppdt.attrs["units"] = "W/m"
    # Unsteadiness of perturbation potential energy
    axisT = Er.eta.get_axis_num(dim="T")
    B["dEp0dt"] = (
        rho0
        * gBaro
        * Er.eta
        * np.gradient(Er.eta, b.time * 3600, axis=axisT)
        * dyF
    ).sum(dim="Y")
    # combine unsteadiness terms
    B["Unsteadiness"] = B.dEkpdt + B.dEppdt + B.dEp0dt
    if steadiness_terms_only:
        return B

    print("Calculating BT/BC conversion term...")
    # (will be tiny but calculate anyways)
    B["Conversion"] = (Er.Conversion * dyF * drF).sum(dim=("Z", "Y"))
    # B["Conversion"] = (["T"], (Er.Conversion * dyF * drF).sum(dim=("Z", "Y")))
    B.Conversion.attrs["name"] = "BT/BC Conversion"
    B.Conversion.attrs["units"] = "W/m"

    print("Calculating horizontal energy advection term...")
    EkpFluxDI = (Er.VC * Er.Ekp * drF * HFacC).sum(dim="Z")
    EppFluxDI = (Er.VC * Er.Epp * drF * HFacC).sum(dim="Z")
    EppSortedFluxDI = (Er.VC * Er.Epp_sorted_rho * drF * HFacC).sum(dim="Z")
    if avg_up_dn:
        B["EkpFluxDivergence"] = EkpFluxDI.isel(Y=ydn).mean(
            dim="Y"
        ) - EkpFluxDI.isel(Y=yup).mean(dim="Y")
    else:
        B["EkpFluxDivergence"] = EkpFluxDI.isel(Y=-1) - EkpFluxDI.isel(Y=0)
    B.EkpFluxDivergence.attrs["name"] = "EK Flux Divergence"
    B.EkpFluxDivergence.attrs["units"] = "W/m"
    if avg_up_dn:
        B["EppFluxDivergence"] = EppFluxDI.isel(Y=ydn).mean(
            dim="Y"
        ) - EppFluxDI.isel(Y=yup).mean(dim="Y")
    else:
        B["EppFluxDivergence"] = EppFluxDI.isel(Y=-1) - EppFluxDI.isel(Y=0)
    B.EppFluxDivergence.attrs["name"] = "APE Flux Divergence"
    B.EppFluxDivergence.attrs["units"] = "W/m"
    # from sorted density
    B["EppSortedFluxDivergence"] = EppSortedFluxDI.isel(
        Y=-1
    ) - EppSortedFluxDI.isel(Y=0)
    B.EppSortedFluxDivergence.attrs["name"] = "APE Flux Divergence"
    B.EppSortedFluxDivergence.attrs["units"] = "W/m"

    print("Calculating vertical energy advection term...")
    # Integrating horizontally along the interface.
    B["EkpVFluxDivergence"] = (
        Er.wp.isel(Z=interface) * Er.Ekp.isel(Z=interface) * dyF
    ).sum(dim="Y")
    B["EppVFluxDivergence"] = (
        Er.wp.isel(Z=interface) * Er.Epp.isel(Z=interface) * dyF
    ).sum(dim="Y")

    print("Calculating horizontal internal wave energy flux term...")
    # hydrostatic
    IWFluxHorizDI = (Er.IWEnergyFluxHoriz * HFacC * drF).sum(dim="Z")
    B["IWFluxHorizDivergence"] = IWFluxHorizDI.isel(Y=-1) - IWFluxHorizDI.isel(
        Y=0
    )
    B.IWFluxHorizDivergence.attrs["name"] = (
        "Horizontal Internal Wave " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence.attrs["units"] = "W/m"
    # from sorted density
    IWSortedFluxHorizDI = (Er.IWEnergyFluxHoriz_sorted_rho * HFacC * drF).sum(
        dim="Z"
    )
    B["IWSortedFluxHorizDivergence"] = IWSortedFluxHorizDI.isel(
        Y=-1
    ) - IWSortedFluxHorizDI.isel(Y=0)
    B.IWSortedFluxHorizDivergence.attrs["name"] = (
        "Horizontal Internal Wave " + "Flux Divergence"
    )
    B.IWSortedFluxHorizDivergence.attrs["units"] = "W/m"
    # non-hydrostatic
    IWFluxHorizDInh = (Er.nhIWEnergyFluxHoriz * HFacC * drF).sum(dim="Z")
    B["IWFluxHorizDivergencenh"] = IWFluxHorizDInh.isel(
        Y=-1
    ) - IWFluxHorizDInh.isel(Y=0)
    B.IWFluxHorizDivergencenh.attrs["name"] = (
        "non-hydrostatic Horizontal " + "Internal Wave Flux Divergence"
    )
    B.IWFluxHorizDivergencenh.attrs["units"] = "W/m"
    # based on local mean profiles
    IWFluxHorizDI_lp = (Er.lp_hwf * HFacC * drF).sum(dim="Z")
    B["IWFluxHorizDivergence_lp"] = IWFluxHorizDI_lp.isel(
        Y=-1
    ) - IWFluxHorizDI_lp.isel(Y=0)
    B.IWFluxHorizDivergence_lp.attrs["name"] = (
        "Horizontal Internal Wave Local Profiles " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence_lp.attrs["units"] = "W/m"
    # based on high-pass filtered time series
    IWFluxHorizDI_hp = (Er.hp_hwf * HFacC * drF).sum(dim="Z")
    B["IWFluxHorizDivergence_hp"] = IWFluxHorizDI_hp.isel(
        Y=-1
    ) - IWFluxHorizDI_hp.isel(Y=0)
    B.IWFluxHorizDivergence_hp.attrs["name"] = (
        "Horizontal Internal Wave High Pass Filter " + "Flux Divergence"
    )
    B.IWFluxHorizDivergence_hp.attrs["units"] = "W/m"

    print("Calculating vertical internal wave energy flux term...")
    # hydrostatic
    B["IWFluxVertDivergence"] = (
        Er.IWEnergyFluxVert.isel(Z=interface) * dyF
    ).sum(dim="Y")
    B.IWFluxVertDivergence.attrs["name"] = (
        "Vertical Internal Wave " + "Flux Divergence"
    )
    B.IWFluxVertDivergence.attrs["units"] = "W/m"
    # hydrostatic (from sorted reference density)
    B["IWSortedFluxVertDivergence"] = (
        Er.IWEnergyFluxVert_sorted_rho.isel(Z=interface) * dyF
    ).sum(dim="Y")
    B.IWSortedFluxVertDivergence.attrs["name"] = (
        "Vertical Internal Wave (sorted)" + "Flux Divergence"
    )
    B.IWSortedFluxVertDivergence.attrs["units"] = "W/m"
    # non-hydrostatic
    B["IWFluxVertDivergencenh"] = (
        Er.nhIWEnergyFluxVert.isel(Z=interface) * dyF
    ).sum(dim="Y")
    B.IWFluxVertDivergencenh.attrs["name"] = (
        "non-hydrostatic Vertical " + "Internal Wave Flux Divergence"
    )
    B.IWFluxVertDivergencenh.attrs["units"] = "W/m"
    # based on local mean profiles
    B["IWFluxVertDivergence_lp"] = (Er.lp_vwf.isel(Z=interface) * dyF).sum(
        dim="Y"
    )
    B.IWFluxVertDivergence_lp.attrs["name"] = (
        "Vertical Internal Wave Local Profiles " + "Flux Divergence"
    )
    B.IWFluxVertDivergence_lp.attrs["units"] = "W/m"
    # based on high-pass filtered time series
    B["IWFluxVertDivergence_hp"] = (Er.hp_vwf.isel(Z=interface) * dyF).sum(
        dim="Y"
    )
    B.IWFluxVertDivergence_hp.attrs["name"] = (
        "Vertical Internal Wave Local Profiles " + "Flux Divergence"
    )
    B.IWFluxVertDivergence_hp.attrs["units"] = "W/m"

    print("Calculating pressure work term due to free ocean surface...")
    PressureWorkSurface = (rho0 * gBaro * Er.eta * Er.VC * HFacC * drF).sum(
        dim="Z"
    )
    B["PressureWorkSurfaceDivergence"] = PressureWorkSurface.isel(
        Y=-1
    ) - PressureWorkSurface.isel(Y=0)

    print("Calculating dissipation term...")
    B["Dissipation"] = (rho0 * Er.eps * drF * dyF * HFacC).sum(dim=("Z", "Y"))
    B.Dissipation.attrs["name"] = "Integrated Dissipation"
    B.Dissipation.attrs["units"] = "W/m"

    print("Calculating bottom drag term...")
    B["BottomDrag"] = (Er.BottomDrag * dyF).sum(dim="Y")
    B.BottomDrag.attrs["name"] = "Integrated Bottom Drag"
    B.BottomDrag.attrs["units"] = "W/m"

    print("Calculating bottom drag [from vels 40m above bottom] term...")
    B["BottomDragAbove"] = (Er.BottomDragAbove * dyF).sum(dim="Y")
    B.BottomDragAbove.attrs["name"] = "Integrated Bottom Drag"
    B.BottomDragAbove.attrs["units"] = "W/m"

    # Combine a few of the terms
    B["IWFDH"] = B.IWFluxHorizDivergence_hp + B.IWFluxHorizDivergencenh
    B["IWFDV"] = B.IWFluxVertDivergence_hp + B.IWFluxVertDivergencenh
    B["RHS"] = (
        B.IWFDH
        + B.IWFDV
        + B.Unsteadiness
        + B.EkpFluxDivergence
        + B.EkpVFluxDivergence
        + B.EppFluxDivergence
        + B.EppVFluxDivergence
        + B.Dissipation
        + B.BottomDrag
    )
    B.attrs["description"] = (
        "Energy budget within a layer defined by\n"
        + "up/downstream distance and isotherm"
    )
    B.attrs["ylims"] = [y1, y2]
    B.attrs["isot"] = isot

    return B


# plotting
def plot_density(b):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharey=True)
    ax[0].plot(b.rhoref, b.Z, label=r"$\rho_{ref}$")
    ax[0].plot(b.rhop.isel(T=-1, Y=1000), b.Z, label=r"$\rho^\prime$")
    # ax[0].plot(b.rho.isel(T=-1,Y=1000),b.Z,label=r'$\rho$')
    ax[0].legend()
    ax[1].plot(b.rhop.isel(T=-1, Y=1000), b.Z, label=r"$\rho^\prime$")
    ax[1].plot(b.rhop.isel(T=-1, Y=1500), b.Z, label=r"$\rho^\prime$")
    ax[1].plot(b.rhop.isel(T=-1, Y=2000), b.Z, label=r"$\rho^\prime$")

    b.rhop.isel(T=-1).plot(ax=ax[2], cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    ax[2].set(xlim=(2e5, 4e5))
    plt.tight_layout()


def plot_pressure_snapshot(b, ti):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    h = ax.pcolormesh(b.dist, -1 * b.Z, b.PP.isel(T=ti), cmap="RdBu_r")
    ax.set(ylim=(5300, 0), ylabel="Depth [m]", xlabel="Distance [km")
    plt.colorbar(h, label=r"p$^\prime$ [N/m$^2$]")
    vmin, vmax = h.get_clim()
    mc = np.max(np.abs([vmin, vmax]))
    h.set_clim(vmin=-1 * mc, vmax=mc)
    ax.fill_between(b.dist, b.Depth, 10000, color="0.2")
    ax.set_title(
        "Pressure perturbation at time={:1.2f} hours".format(
            b.time.isel(T=ti).values.tolist()
        )
    )


def plot_velocity_snapshot(b, ti):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    h = ax.pcolormesh(b.dist, -1 * b.Z, b.v.isel(T=ti), cmap="RdBu_r")
    ax.set(ylim=(5300, 0), ylabel="Depth [m]", xlabel="Distance [km")
    plt.colorbar(h, label=r"v$^\prime$ [m/s]")
    vmin, vmax = h.get_clim()
    mc = np.max(np.abs([vmin, vmax]))
    h.set_clim(vmin=-1 * mc, vmax=mc)
    ax.fill_between(b.dist, b.Depth, 10000, color="0.2")
    ax.set_title(
        "Baroclinic velocity at time={:1.2f} hours".format(
            b.time.isel(T=ti).values.tolist()
        )
    )


def plot_IWfluxH_snapshot(b, E, ti):
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    # ax = gv.plot.axstyle(ax)
    fig, ax = gv.plot.quickfig(w=8, h=3)
    cmap = "PuOr"
    h = ax.pcolormesh(
        b.dist,
        -1 * b.Z,
        E.IWEnergyFluxHoriz.isel(T=ti),
        cmap=cmap,
    )
    ax.fill_between(b.dist, b.Depth, 10000, color="0.6")
    ax.set(
        ylim=(5300, 4000),
        xlim=(-50, 180),
        ylabel="Depth [m]",
        xlabel="Distance [km]",
    )
    plt.colorbar(
        h, label=r"v$^\prime$ p$^\prime$ [W/m$^2$]", shrink=0.8, aspect=50
    )
    vmin, vmax = h.get_clim()
    vmin, vmax = -50, 50
    mc = np.max(np.abs([vmin, vmax]))
    h.set_clim(vmin=-1 * mc, vmax=mc)
    ax.text(
        -40,
        5250,
        "t = {:1.2f} hours".format(b.time.isel(T=ti).values.tolist()),
        color="w",
    )
    # ax.set_title(
    #     "Horizontal internal wave energy flux at time {:1.2f} hours".format(b.time.isel(T=ti).values.tolist())
    # )


def plot_IWfluxH_snapshot_and_depth_integrated_time_series(b, E, ti):
    fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    for axi in [ax, ax2]:
        axi = gv.plot.axstyle(axi, fontsize=12)
    cmap = "PuOr"
    h = ax.pcolormesh(
        b.dist,
        -1 * b.Z,
        E.IWEnergyFluxHoriz.isel(T=ti),
        cmap=cmap,
    )
    ax.fill_between(b.dist, b.Depth, 10000, color="0.4")
    ax.set(ylim=(5300, 4000), xlim=(-50, 180), ylabel="depth [m]", xlabel="")
    cb = plt.colorbar(h, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cb.set_label(label=r"v$^\prime$ p$^\prime$ [W/m$^2$]", size="large")
    cb.ax.tick_params(labelsize="large")
    vmin, vmax = h.get_clim()
    vmin, vmax = -50, 50
    mc = np.max(np.abs([vmin, vmax]))
    h.set_clim(vmin=-1 * mc, vmax=mc)
    ax.text(
        -40,
        5250,
        "t = {:1.2f} hours".format(b.time.isel(T=ti).values.tolist()),
        color="w",
    )
    # depth-integrated time series
    iwflux = E.IWEnergyFluxHoriz.swap_dims({"Y": "dist", "T": "time"}).where(
        E.Z < -3000, drop=True
    )
    iwflux = iwflux.where(~np.isnan(iwflux.data), other=0)
    intiwflux = iwflux.integrate(dim="Z") / 1e3
    h2 = intiwflux.plot(ax=ax2, cmap="YlGnBu", robust=True, add_colorbar=False)
    cb2 = plt.colorbar(
        h2,
        ax=ax2,
        shrink=0.8,
        aspect=30,
        pad=0.02,
    )
    cb2.set_label(label=r"$\int v^\prime p^\prime\ dz$ [kW/m]", size="large")
    cb2.ax.tick_params(labelsize="large")
    ax2.set(xlim=(-50, 180), ylabel="model time [h]", xlabel="distance [km]")
    ax2.invert_yaxis()

    for axi, label in zip([ax, ax2], ["a", "b"]):
        gv.plot.annotate_corner(
            label,
            axi,
            addy=0.055,
            addx=-0.015,
            fs=12,
            col="k",
            quadrant=1,
            fw="bold",
            background_circle="w",
        )


def plot_IWfluxV_snapshot(b, E, ti):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax = gv.plot.axstyle(ax)

    h = ax.pcolormesh(
        b.dist, -1 * b.Z, E.IWEnergyFluxVert.isel(T=ti), cmap="RdBu_r"
    )
    clevels = [0.76, 0.8, 0.9, 1.0, 1.1]
    hc = ax.contour(
        b.dist,
        -1 * b.Z,
        b.th.isel(T=ti),
        levels=clevels,
        colors="k",
        linewidths=0.5,
    )
    ax.set(
        ylim=(5300, 3500),
        ylabel="Depth [m]",
        xlim=(-30, 50),
        xlabel="Distance [km",
    )
    manual_positions = list(
        zip([40, 40, 40, 40, 40], [3900, 4200, 4500, 4600, 4700])
    )
    plt.clabel(
        hc,
        manual=manual_positions,
        inline=True,
        inline_spacing=2,
        fmt="%1.2f",
        fontsize=7,
    )
    ax.fill_between(b.dist, b.Depth, 10000, color="0.2")
    plt.colorbar(h, label=r"w$^\prime$ p$^\prime$ [W/m$^2$]")
    vmin, vmax = h.get_clim()
    mc = np.max(np.abs([vmin, vmax]))
    # set the clim, but don't go to the extremes
    h.set_clim(vmin=-1 * (mc - mc / 5), vmax=mc - mc / 5)
    ax.set_title(
        "Vertical internal wave energy flux at time"
        + "={:1.2f} hours".format(b.time.isel(T=ti).values.tolist())
    )
    return h


def plot_IWfluxV_snapshot_large_region(b, E, ti):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    ax = gv.plot.axstyle(ax)

    h = ax.pcolormesh(
        b.dist,
        -1 * b.Z,
        E.IWEnergyFluxVert.isel(T=ti),
        # cmap="RdBu_r",
        cmap=cmocean.cm.delta,
        norm=mpl.colors.SymLogNorm(
            linthresh=0.02, linscale=1.5, vmax=20, base=10
        ),
    )
    clevels = [0.76, 0.8, 0.9, 1.0, 1.1]
    hc = ax.contour(
        b.dist,
        -1 * b.Z,
        b.th.isel(T=ti),
        levels=clevels,
        colors="k",
        linewidths=0.5,
    )
    ax.set(
        ylabel="Depth [m]",
        xlabel="Distance [km",
    )
    manual_positions = list(
        zip([40, 40, 40, 40, 40], [3900, 4200, 4500, 4600, 4700])
    )
    plt.clabel(
        hc,
        manual=manual_positions,
        inline=True,
        inline_spacing=2,
        fmt="%1.2f",
        fontsize=7,
    )
    ax.fill_between(b.dist, b.Depth, 10000, color="0.2")
    plt.colorbar(h, label=r"w$^\prime$ p$^\prime$ [W/m$^2$]")
    vmin, vmax = h.get_clim()
    mc = np.max(np.abs([vmin, vmax]))
    # set the clim, but don't go to the extremes
    h.set_clim(vmin=-1 * (mc - mc / 5), vmax=mc - mc / 5)
    ax.set_title(
        "Vertical internal wave energy flux at time"
        + "={:1.2f} hours".format(b.time.isel(T=ti).values.tolist())
    )
    ax.set(xlim=(-100, 100), ylim=(5300, 1000))
    return h, ax


def plot_initial_stratification(b, zlim=4000):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    gv.plot.axstyle(ax)
    tmp = b.th.where(b.Z < -zlim, drop=True).isel(T=0)
    h = ax.pcolormesh(
        tmp.dist, -tmp.Z, np.ma.masked_equal(tmp.values, 0), cmap="Spectral"
    )
    ax.contour(
        tmp.dist,
        -tmp.Z,
        np.ma.masked_equal(tmp.values, 0),
        levels=np.arange(0.7, 1.15, 0.05),
        colors="k",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.colorbar(h, ax=ax)
    ax.fill_between(b.dist, b.Depth, 10000, color="0.2")
    ax.set(xlim=(-20, 35), ylim=(5300, zlim))
    ax.grid(True)
    ax.set(ylabel="Depth [m]", xlabel="Distance [km]")


def plot_snapshot(b: xr.Dataset, ti: int, zlim: float = 3000):
    """Plot model snapshot.

    Show four panels with (1) temperature, (2) horizontal velocity, (3)
    vertical velocity and (4) turbulent dissipation.

    Parameters
    ----------
    b : xarray.Dataset
        Model data.
    ti : int
        Index into time vector.
    zlim : float, optional
        Upper depth limit in meters. Defaults to 3000.
    """
    b = b.isel(T=ti)
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 15), sharex=True)
    for axi in ax:
        gv.plot.axstyle(axi, ticks="in", fontsize=11)

    tmpth = b.th.where(b.Z < -zlim, drop=True)
    h = ax[0].pcolormesh(
        tmpth.dist,
        -tmpth.Z,
        np.ma.masked_equal(tmpth.values, 0),
        cmap="Spectral_r",
    )
    colbaropts = dict(pad=0.02)

    def contour_th(ax, tmpth):
        ax.contour(
            tmpth.dist,
            -tmpth.Z,
            np.ma.masked_equal(tmpth.values, 0),
            levels=np.arange(0.7, 1.55, 0.05),
            colors="k",
            linewidths=0.5,
            alpha=0.5,
        )

    def contour_interface(ax, tmpth):
        ax.contour(
            tmpth.dist,
            -tmpth.Z,
            np.ma.masked_equal(tmpth.values, 0),
            levels=[0.8, 0.9],
            colors="k",
            linewidths=1.5,
            alpha=0.5,
        )

    contour_th(ax[0], tmpth)
    contour_interface(ax[0], tmpth)
    plt.colorbar(h, ax=ax[0], label=r"$\theta$ [$^{\circ}$C]", **colbaropts)

    tmp = b.VC.where(b.Z < -zlim, drop=True)
    h1 = ax[1].pcolormesh(
        tmp.dist,
        -tmp.Z,
        np.ma.masked_equal(tmp.values, 0),
        vmin=-0.4,
        vmax=0.4,
        cmap="RdBu_r",
    )
    contour_th(ax[1], tmpth)
    contour_interface(ax[1], tmpth)
    plt.colorbar(h1, ax=ax[1], extend="max", label="v [m/s]", **colbaropts)

    tmp = b.WC.where(b.Z < -zlim, drop=True)
    h2 = ax[2].pcolormesh(
        tmp.dist,
        -tmp.Z,
        np.ma.masked_equal(tmp.values, 0),
        vmin=-0.05,
        vmax=0.05,
        cmap="RdBu_r",
    )
    contour_th(ax[2], tmpth)
    contour_interface(ax[2], tmpth)
    plt.colorbar(h2, ax=ax[2], extend="max", label="w [m/s]", **colbaropts)

    tmp = b.eps.where(b.Z < -zlim, drop=True)
    h3 = ax[3].pcolormesh(
        tmp.dist,
        -tmp.Z,
        np.log10(np.ma.masked_equal(tmp.values, 0)),
        vmin=-10,
        vmax=-5,
        cmap=cmocean.cm.speed,
    )
    contour_th(ax[3], tmpth)
    contour_interface(ax[3], tmpth)
    plt.colorbar(
        h3, ax=ax[3], extend="both", label=r"$\epsilon$ [W/kg]", **colbaropts
    )

    for axi in ax:
        axi.fill_between(b.dist, b.Depth, 10000, color="0.2")
        axi.set(xlim=(0, 35), ylim=(5300, zlim))
        axi.grid(True)
    # ax[1].set(ylabel="Depth [m]")
    plt.annotate(
        "Depth [m]",
        xy=(0.04, 0.49),
        xycoords="figure fraction",
        ha="center",
        rotation=90,
        fontsize=12,
    )
    ax[3].set(xlabel="Distance [km]")
    gv.plot.subplotlabel(ax)


def plot_energy_budget(B, plotall=True):
    """
    Plot various terms of the baroclinic energy equation.

    Parameters
    ----------
    B : xarray Dataset
        Output of energy_budget_box_Z()

    Notes
    -----
    Showing the following panels:

      - Unsteadiness, dEkpdt, dEppdt
      - IWFluxes
      - EK & APE flux divergence
      - all

    """
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 16), sharex=True)

    ax[0].plot(B.time, B.Unsteadiness / 1e3, label="Unsteadiness")
    ax[0].plot(B.time, B.dEp0dt / 1e3, label="dEp0dt")
    ax[0].plot(B.time, B.dEkpdt / 1e3, label="dEkpdt")
    ax[0].plot(B.time, B.dEppdt / 1e3, label="dEppdt")

    ax[1].plot(
        B.time,
        B.IWFluxHorizDivergence_hp / 1e3,
        label="Horizontal IW Flux Divergence",
    )
    ax[1].plot(
        B.time,
        B.IWFluxHorizDivergencenh / 1e3,
        label="nh Horizontal IW Flux Divergence",
    )
    ax[1].plot(
        B.time,
        B.IWFluxVertDivergence_hp / 1e3,
        label="Vertical IW Flux Divergence",
    )
    ax[1].plot(
        B.time,
        B.IWFluxVertDivergencenh / 1e3,
        label="nh Vertical IW Flux Divergence",
    )

    ax[2].plot(B.time, B.EkpFluxDivergence / 1e3, label="EK Flux Divergence H")
    ax[2].plot(B.time, B.EkpVFluxDivergence / 1e3, label="EK Flux Divergence V")
    ax[2].plot(B.time, B.EppFluxDivergence / 1e3, label="APE Flux Divergence H")
    ax[2].plot(
        B.time, B.EppVFluxDivergence / 1e3, label="APE Flux Divergence V"
    )

    ax[3].plot(B.time, B.Unsteadiness / 1e3, label="Unsteadiness")
    ax[3].plot(
        B.time,
        (B.EkpFluxDivergence + B.EkpVFluxDivergence) / 1e3,
        label="EK Flux Divergence",
    )
    ax[3].plot(
        B.time,
        (B.EppFluxDivergence + B.EppVFluxDivergence) / 1e3,
        label="APE Flux Divergence",
    )
    ax[3].plot(B.time, B.Dissipation / 1e3, label="Integrated Dissipation")
    ax[3].plot(B.time, B.BottomDrag / 1e3, label="Integrated Bottom Drag")

    if plotall:
        ax[3].plot(B.time, B.IWFDH / 1e3, label="Horiz IW Flux Divergence")
        ax[3].plot(B.time, B.IWFDV / 1e3, label="Vert IW Flux Divergence")
        ax[3].plot(B.time, B.RHS / 1e3, label="all terms", color="k")

    for axi in ax:
        axi.grid(True)
        axi.legend(loc=(1.01, 0))
        axi.set(ylabel="dE/dt [kW/m]")
    ax[-1].set(xlabel="time [hrs]")


def plot_steadiness_terms(B):
    """
    Plot steadiness terms in the baroclinic energy equation.

    Parameters
    ----------
    B : xarray.Dataset
        Output of :class:`nslib.model.energy_budget_box` or :class:`nslib.model.energy_budget_layer`.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True)
    ax = gv.plot.axstyle(ax)
    ax.plot(
        B.time,
        B.dEp0dt / 1e3,
        label=r"$\frac{\partial}{\partial t} E_{p0}$",
        color="C7",
    )
    ax.plot(
        B.time,
        B.dEppdt / 1e3,
        label=r"$\frac{\partial}{\partial t} E_p^\prime$",
        color="C4",
    )
    ax.plot(
        B.time,
        B.dEkpdt / 1e3,
        label=r"$\frac{\partial}{\partial t} E_k^\prime$",
        color="C0",
    )
    ax.plot(
        B.time,
        B.Unsteadiness / 1e3,
        label=r"$\frac{\partial}{\partial t} (E_k^\prime + E_p^\prime + E_{p0})$",
        color="C3",
    )
    ax.legend(fontsize=13)
    anncol = "0.2"
    ax.vlines(100, 1.5, 2, color=anncol)
    ax.vlines(150, 1.5, 2, color=anncol)
    ax.annotate(
        "",
        xy=(100, 1.75),
        xycoords="data",
        xytext=(150, 1.75),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="<->",
            color=anncol,
        ),
    )
    ax.annotate("analysis\nperiod", xy=(125, 2), xycoords="data", ha="center")
    ax.set(xlabel="model time [h]", ylabel="dE/dt [kW/m]")


# helper plot functions
def plot_center(dy=50, zz=4000, ax=0):
    if ax == 0:
        ax = plt.gca()
    ax.set(xlim=(-dy, dy), ylim=(5300, zz))


def plot_seafloor(b):
    plt.fill_between(b.dist, b.Depth, 10000, color="0.2")


def plotsill(b, var, ti=0, **kwargs):
    if "Y" in b.dims:
        b = b.swap_dims({"Y": "dist"})
    if "T" in b.dims:
        b = b.swap_dims({"T": "time"})
    fig, ax = gv.plot.quickfig()
    b[var].isel(time=ti).plot(ax=ax, **kwargs)
    ax.set(xlim=(-10, 40), ylim=(-5200, -4200))


# constants
gravity = 9.81
r"""Gravity

.. math::

    g=9.81
"""

gBaro = 0.98
r"""(Barotropic) gravity for free surface (eta).

.. math::

    gBaro=0.98
"""

rho0 = 9.998000000000000e02  # from STDOUT, e.g.: grep -A 1 'rho' STDOUT.0000
"""Model reference density `rhoNil`."""

nuh = 1e-4
r"""Horizontal eddy viscosity `viscAh`

.. math::

    \nu_H=10^{-4} \mathrm{m}^2/\mathrm{s}
"""

kappah = 1e-4
r"""Horizontal diffusivity of heat (and thereby mass with the LES, `diffKhT`):

.. math::

    \kappa_H=10^{-4} \mathrm{m}^2/\mathrm{s}
"""

nuv = 1e-5
r"""Vertical eddy viscosity `viscAv`

.. math::

    \nu_H=10^{-5} \mathrm{m}^2/\mathrm{s}
"""

kappav = 1e-5
r"""Vertical diffusivity of heat `diffKvT`

.. math::

    \kappa_V=10^{-5} \mathrm{m}^2/\mathrm{s}
"""

Cd = 1e-3
r"""Bottom drag coefficient

.. math::

    C_d = 1 \times 10^{-3}
"""
