#!/usr/bin/env python
# coding: utf-8

r"""Calculate local mean profiles and wave fluxes based on these for the 2D
MITgcm simulations of the Samoan Passage Northern Sill overflow.

Quick overview: For a given model run, calculate primed quantities based on
local (spatial means), just as we do for the observations. Do this calculation
only for a sub-region centered in depth and horizontal extent around the sill
to save computation cost. From the primed quantities, calculate internal wave
fluxes $v^\prime p^\prime$ and $w^\prime p^\prime$ and integrate these. See
:ref:`wave_fluxes` for a comparison of results based on using various window
sizes.

The time series of spatial means are saved for quicker re-computation of the
products derived thereof.

The main functions are :fun:`extract_subregion`,
:fun:`calculate_spatial_means`, :fun:`calculate_primed_quantities`,
:fun:`calculate_fluxes` and :fun:`integrate_vertical_iw_flux`.

All calculations can be run with :class:`small_scale_flux_calculations`.

Also added internal wave flux calculations based on high-pass filtered model time series of pressure and velocity under :class:`hp_flux_calculations`.
"""


from pathlib import Path
from box import Box
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xarray as xr
import gvpy as gv
import pickle
import scipy as sp
from tqdm import tqdm  # a little progress bar
import cmocean

import nslib as nsl


def extract_subregion(b, distmin=-10, distmax=50, depthmin=-3800, timeind=None):
    """Extract model subregion.

    Parameters
    ----------
    b : xr.Dataset
        Model data.
    distmin : float, optional
        Minimum horizontal coordinate for selected subregion in km. Defaults to
        -10.
    distmax : float, optional
        Maximum horizontal coordinate for selected subregion in km. Defaults to
        50.
    depthmin : float, optional
        Minimum z coordinate (z decreases downwards in the ocean) for selected
        subregion. Defaults to -3800.
    timeind : int or range(), optional
        Index or indices for time subset. Whole time series if not provided.

    Returns
    -------
    subset : xr.Dataset
        Model subregion (and time range if provided as timeind).
    """

    # Use time and distance instead of model units
    a = b.swap_dims({"T": "time", "Y": "dist"})

    if timeind is None:
        timeind = range(b.time.shape[0])

    # Extract a number of variables
    hfacc = a.HFacC.where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    bdep = a.Depth.where((a.dist > distmin) & (a.dist < distmax), drop=True)
    drf = a.drF.where(a.Z < depthmin).copy()
    dyf = a.dyF.where((a.dist > distmin) & (a.dist < distmax), drop=True).copy()

    v = a.v.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    WC = a.WC.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    wp = a.wp.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    PP = a.PP.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    q = a.q.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    rho = a.rho.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    th = a.th.isel(time=timeind).where(
        (a.dist > distmin) & (a.dist < distmax) & (a.Z < depthmin), drop=True
    )
    # combine all in one output dataset
    subset = xr.merge(
        [v, WC, wp, PP, q, rho, th, hfacc, bdep, drf, dyf], compat="override"
    )
    subset = subset.dropna("Z", how="all")
    return subset


def calculate_spatial_means(
    b,
    savedir=None,
    recalc=False,
    timeind=None,
    saveresult=False,
    window_size=5e3,
):
    """Calculate local mean profiles for model data.

    Parameters
    ----------
    b : xr.Dataset
        Model data.
    savedir : Path, optional
        Path to save or read results from. Defaults to northern sill analysis
        data directory if none provided.
    recalc : bool, optional
        Recalculate background means. Defaults to False since this can take
        some time for a long model time series.
    timeind : int or range(), optional
        Index or indices for time subset. Whole time series if not provided.
    saveresult : bool, optional
        Save result to savedir. Defaults to False.
    window_size : int, optional
        Window size for rolling window mean in meters. Defaults to 5e3.

    Returns
    -------
    ms : xr.Dataset
        Mean velocities and density.
    """
    cfg = nsl.io.load_config()
    savedir = cfg.path.output
    file = savedir.joinpath("model_spatial_means.nc")
    if file.exists() is False:
        recalc = True
    if recalc:
        s = extract_subregion(b, timeind=timeind)
        mw = mean_w(s, window_size=window_size)
        mv = mean_v(s, window_size=window_size)
        mrho = mean_rho(s, window_size=window_size)
        ms = xr.merge([mv, mw, mrho])
        ms.to_netcdf(file)
    else:
        ms = xr.open_dataset(file)
        file.touch()
    return ms


def calculate_primed_quantities(subset, ms):
    """Calculate primed quantities for model subset based on local spatial means.

    Parameters
    ----------
    subset : xr.Dataset
        Model subset selected with `extract_subregion`.
    ms : xr.Dataset
        Model local spatial means calculated with `calculate_spatial_means`.

    Returns
    -------
    p : xr.Dataset
        Model deviations from local spatial means.
    """

    wp = calculate_wp(ms.mw, subset)
    vp = calculate_vp(ms.mv, subset)
    rhop, pp = calculate_pp(ms.mrho, subset)
    p = xr.merge([vp, wp, pp, rhop])
    return p


def integrate_vertical_iw_flux_along_isotherm(vwf, subset, isot):
    """Integrate model vertical internal wave flux horizontally along an
    isotherm.

    Parameters
    ----------
    vwf : xr.DataArray
        Model vertical internal wave flux calculated with `calculate_fluxes`.
    subset : xr.Dataset
        Model subregion (and time range if provided as timeind).
    isot : float
        Isotherm to integrate along.

    Returns
    -------
    ivwf : xr.DataArray
        Horizontally integrated vertical wave flux.
    """
    interface = subset.th.where(subset.th < isot).argmax(dim="Z")
    return (vwf.isel(Z=interface) * subset.dyF).sum(dim="dist")


def integrate_vertical_iw_flux(vwf, subset, intmin=0.0, intmax=17.0):
    """Horizontally integrate model vertical wave flux along isotherms.

    Isotherms range from 0.66°C to 1.2°C. Default horizontal integration range
    is km 0 to km 17.

    Parameters
    ----------
    vwf : xr.DataArray
        Model vertical internal wave flux calculated with `calculate_fluxes`.
    subset : xr.Dataset
        Model subregion (and time range if provided as timeind).
    intmin : float
        Minimum horizontal integration limit.
    intmax : float
        Maximum horizontal integration limit.

    Returns
    -------
    vwfi : xr.DataArray
        Integrated vertical wave flux.
    """

    upstream = intmin
    downstream = intmax
    print(f'integrating from {intmin} to {intmax}')
    vwfs = vwf.where((vwf.dist > upstream) & (vwf.dist < downstream), drop=True)
    subsubset = subset.where(
        (subset.dist > upstream) & (subset.dist < downstream), drop=True
    )
    isotherms = np.linspace(0.66, 1.2, 50)
    vwf_integrated = np.array(
        [
            integrate_vertical_iw_flux_along_isotherm(vwfs, subsubset, isot)
            for isot in isotherms
        ]
    )
    if 'time' in vwf.dims:
        vwfi = xr.DataArray(
            data=vwf_integrated,
            dims=("isot", "time"),
            coords={"isot": isotherms, "time": subset.time},
        )
    else:
        vwfi = xr.DataArray(
            data=vwf_integrated,
            dims=("isot"),
            coords={"isot": isotherms},
        )
    return vwfi


def calculate_fluxes(p):
    """Calculate model wave fluxes based on local spatial means.

    Parameters
    ----------
    p : xr.Dataset
        Model deviations from local spatial means.

    Returns
    -------
    fluxes : xr.Dataset
        Model wave fluxes.
    """

    vwf = calculate_vertical_wave_flux(p)
    hwf = calculate_horizontal_wave_flux(p)
    momf = calculate_momentum_flux(p)
    fluxes = xr.merge([vwf, hwf, momf])
    return fluxes


def calculate_vertical_wave_flux(p):
    vwf = p.wp * p.pp
    vwf.name = "vwf"
    return vwf


def calculate_horizontal_wave_flux(p):
    hwf = p.wp * p.pp
    hwf.name = "hwf"
    return hwf


def calculate_momentum_flux(p):
    momf = p.vp * p.wp
    momf.name = "momf"
    return momf


def mean_w(subset, window_size=5e3, verbose=False):
    zxh, hxz = transformation_matrices(subset.WC, subset.Depth)
    if "time" in subset.WC.dims:
        if verbose:
            print("w - iterating over all time steps")
        tmp = []
        for g, wc in tqdm(subset.WC.groupby("time")):
            tmp.append(
                calculate_background_mean(
                    wc,
                    zxh,
                    hxz,
                    window_size=window_size,
                )
            )
        mw = xr.concat(tmp, dim="time")
    else:
        if verbose:
            print("w - only one time step")
        mw = calculate_background_mean(
            subset.WC,
            zxh,
            hxz,
            window_size=window_size,
        )
    mw.name = "mw"
    return mw


def mean_v(subset, window_size=5e3, verbose=False):
    zxh, hxz = transformation_matrices(subset.v, subset.Depth)
    if "time" in subset.v.dims:
        if verbose:
            print("v - iterating over all time steps")
        tmp = []
        for g, vc in tqdm(subset.v.groupby("time")):
            tmp.append(
                calculate_background_mean(
                    vc,
                    zxh,
                    hxz,
                    window_size=window_size,
                )
            )
        mv = xr.concat(tmp, dim="time")
    else:
        if verbose:
            print("v - only one time step")
        mv = calculate_background_mean(
            subset.v,
            zxh,
            hxz,
            window_size=window_size,
        )
    mv.name = "mv"
    return mv


def mean_rho(subset, window_size=5e3, verbose=False):
    zxh, hxz = transformation_matrices(subset.rho, subset.Depth)
    if "time" in subset.rho.dims:
        if verbose:
            print("rho - iterating over all time steps")
        tmp = []
        for g, rho in tqdm(subset.rho.groupby("time")):
            tmp.append(
                calculate_background_mean(
                    rho,
                    zxh,
                    hxz,
                    transform_to_hab=False,
                    window_size=window_size,
                )
            )
        mrho = xr.concat(tmp, dim="time")
    else:
        if verbose:
            print("rho - only one time step")
        mrho = calculate_background_mean(
            subset.rho,
            zxh,
            hxz,
            transform_to_hab=False,
            window_size=window_size,
        )
    mrho.name = "mrho"
    return mrho


def calculate_wp(mw, subset):
    wp = subset.WC - mw
    wp.name = "wp"
    return wp


def calculate_vp(mv, subset):
    vp = subset.v - mv
    vp.name = "vp"
    return vp


def calculate_pp(mrho, subset):
    rhop = subset.rho - mrho
    rhop.name = "rhop"
    pp = (
        (rhop * subset.drF * 9.81).isel(Z=slice(-1, 0, -1)).cumsum(dim="Z")
    ) * subset.HFacC.round()
    pp.name = "pp"
    return rhop, pp


def calculate_background_mean(
    x, zxh, hxz, window_size=5e3, hfacc=None, transform_to_hab=True
):
    """Calculate model local mean profiles.

    This is based on the following steps:

    - Interpolate to regular 10m grid in the horizontal and the vertical
    - Transform to height above bottom coordinates (optional)
    - Calculate windowed mean
    - Transform back to depth coordinates
    - Interpolate back to original model grid

    Parameters
    ----------
    x : xr.DataArray
        Model field in hab coordinates
    zxh : xr.DataArray
        z(x, hab)
    hxz : xr.DataArray
        hab(x, z)
    window_size : int, optional
        Window size for rolling window mean in meters. Defaults to 5e3.
    hfacc : xr.DataArray, optional
        Matrix with partial bottom cells. Applied in a non-optimal way to the
        interpolation if supplied.
    transform_to_hab : bool, optional
        Transform model field to height above bottom coordinates for
        calculating local mean profile. Defaults to True.
    """
    xi = interp_to_regular_grid(x, hfacc)
    # Transform to height above bottom
    if transform_to_hab:
        xih = z2hab(xi, zxh, hxz)
    else:
        xih = xi
    # Moving average
    mxih = xih.rolling(
        dist=int(window_size / 10), center=True, min_periods=20
    ).mean()
    # Transform back to depth coordinates
    if transform_to_hab:
        mxiz = hab2z(mxih, zxh, hxz)
    else:
        mxiz = mxih
    # Interpolate back to model grid
    mx = mxiz.interp_like(x)
    return mx


def interp_to_regular_grid(x, hfacc=None):
    """Interpolate model field to a regularly spaced grid
    with 10m resolution in both the horizontal and the vertical.

    Parameters
    ----------
    x : xr.DataArray
        Model field
    hfacc : xr.DataArray, optional
        Matrix with partial bottom cells. Applied in a non-optimal way to the
        interpolation if supplied.

    Returns
    -------
    xi : xr.DataArray
        Interpolate model data
    """
    distn = np.arange(x.dist.min(), x.dist.max(), 10 / 1e3)
    zn = np.arange(x.Z.min(), x.Z.max(), 10)
    xi = x.interp(dist=distn, Z=zn)
    if hfacc is not None:
        hfi = hfacc.interp(Z=zn, dist=distn)
        mask = hfi.where(hfi < 0.5, other=1)
        mask = mask.where(hfi >= 0.5, other=0)
    else:
        mask = np.ones_like(xi)
    return xi * mask


def transformation_matrices(x, bdep):
    """Generate transformation matrices between depth and height above
    bottom coordinate system.

    Parameters
    ----------
    x : xr.DataArray
        Model field
    bdep : xr.DataArray
        Bottom depth

    Returns
    -------
    zxh : xr.DataArray
        z(x, hab)
    hxz : xr.DataArray
        hab(x, z)
    """
    # interpolate x to the regular grid to obtain grid parameters
    xi = interp_to_regular_grid(x)
    distn = xi.dist.data
    bdepi = bdep.interp(dist=distn).drop("Y")
    # generate a common depth matrix
    Z = xr.DataArray(
        np.tile(xi.Z.data, [xi.dist.shape[0], 1]),
        dims=["dist", "Z"],
        coords={"dist": xi.dist.data, "Z": xi.Z.data},
    )
    hxz = bdepi + Z
    # new coordinates
    habcoord = np.arange(0, xi.Z.shape[0] * 10, 10)
    distcoord = xi.dist.data.copy()
    # generate empty array
    zxh = xr.DataArray(
        np.full_like(Z.T, np.nan),
        dims=["hab", "dist"],
        coords={"hab": habcoord, "dist": distcoord},
    )
    # interpolate
    for i in range(hxz.dist.shape[0]):
        zxh.data[:, i] = sp.interpolate.interp1d(
            hxz[i, :], hxz.Z.data, bounds_error=False
        )(habcoord)
    return zxh, hxz


def z2hab(x, zxh, hxz):
    """Transform model field from depth to hab coordinates.

    Parameters
    ----------
    x : xr.DataArray
        Model field in z coordinates
    zxh : xr.DataArray
        z(x, hab)
    hxz : xr.DataArray
        hab(x, z)

    Returns
    -------
    xh : xr.DataArray
        Model field in hab coordinates
    """
    newx = xr.DataArray(
        x.dist.data, dims=["dist"], coords={"dist": x.dist.data}
    )
    # the interpolation doesn't like nan's in the interpolation matrix. we
    # could have eliminated them by providing more data when generating
    # z(x,h) but for now we'll just replace them with large numbers. here it
    # doesn't matter because we also have no data to interpolate where the
    # interpolation matrix is nan.
    newz = zxh.where(~np.isnan(zxh), other=9999)
    # interpolate to hab coordinate system
    xh = x.interp(dist=newx, Z=newz).drop("Z")
    return xh


def hab2z(x, zxh, hxz):
    """Transform model field from hab to depth coordinates.

    Parameters
    ----------
    x : xr.DataArray
        Model field in hab coordinates
    zxh : xr.DataArray
        z(x, hab)
    hxz : xr.DataArray
        hab(x, z)

    Returns
    -------
    xz : xr.DataArray
        Model field in z coordinates
    """
    newx = xr.DataArray(
        x.dist.data, dims=["dist"], coords={"dist": x.dist.data}
    )
    # the interpolation doesn't like nan's in the interpolation matrix. we
    # could have eliminated them by providing more data when generating
    # z(x,h) but for now we'll just replace them with large numbers. here it
    # doesn't matter because we also have no data to interpolate where the
    # interpolation matrix is nan.
    newz = hxz.where(~np.isnan(hxz), other=9999)
    # interpolate to z coordinate system
    xz = x.interp(dist=newx, hab=newz)
    return xz


def small_scale_flux_calculations(recalc=False):
    """Calculate model internal wave fluxes based on local spatial means.

    Parameters
    ----------
    recalc : bool, optional
        Recalculate spatial means. Time intensive. Defaults to False.

    Notes
    -----
    Saves `cfg.model.output.small_scale_fluxes` and
    `cfg.model.output.small_scale_vwf_integrated`.
    """
    cfg = nsl.io.load_config()

    # Read model data
    b = xr.open_dataset(cfg.model.output.data)
    # Calculate spatial means for subregion (or load existing)
    s = extract_subregion(b)
    ms = calculate_spatial_means(b, recalc=recalc)
    # Calculate wave fluxes based on spatial means
    p = calculate_primed_quantities(s, ms)
    fluxes = calculate_fluxes(p)
    # Save fluxes to netcdf file
    nsl.io.close_nc(cfg.model.output.small_scale_fluxes)
    fluxes.to_netcdf(cfg.model.output.small_scale_fluxes)
    # Integrate vertical wave fluxes
    intmin = cfg.parameters.model.integration_horizontal_min
    intmax = cfg.parameters.model.integration_horizontal_max
    vwfi = integrate_vertical_iw_flux(fluxes.vwf, s, intmin, intmax)
    # Save integrated fluxes
    nsl.io.close_nc(cfg.model.output.small_scale_vwf_integrated)
    vwfi.to_netcdf(cfg.model.output.small_scale_vwf_integrated)


def hp_filter(b, var, cutoff_period):
    """High pass filter time series of model variable.

    Parameters
    ----------
    b : xr.Dataset
        Model data
    var : str
        Variable
    cutoff_period : float
        Cutoff period in hours.

    Returns
    -------
    b : xr.Dataset
        Model data

    Notes
    -----
    Adds variable hp_[var] to the dataset.
    """

    timeax = 0
    x = b[var]
    # replace nan's with zeros for filtering process
    x = x.where(~np.isnan(x), other=0)
    x = sp.signal.detrend(x, axis=timeax)
    # high pass filter. model sampling frequency is 4 per hour.
    tmp = gv.signal.highpassfilter(
        x, highcut=1 / cutoff_period, fs=4, order=3, axis=timeax
    )
    hpvarname = "hp_" + var
    b[hpvarname] = (["time", "Z", "dist"], tmp)
    putnan = np.all(tmp == 0, axis=0)
    b[hpvarname] = b[hpvarname].where(~putnan, other=np.nan)
    return b


def hp_flux_calculations(hp_cutoff_period=12):
    """Calculate model internal wave fluxes based on high pass-filtered time series.

    Parameters
    ----------
    hp_cutoff_period : float
        Cutoff period in hours.

    Notes
    -----
    Save results to `cfg.model.output.hp_fluxes` and cfg.model.output.hp_vwf_integrated.
    """
    cfg = nsl.io.load_config()
    # Read model data
    bb = xr.open_dataset(cfg.model.output.data)
    print('extract subregion')
    b = extract_subregion(bb)
    # High-pass filter pressure (hydrostatic and nh) and velocity
    b = hp_filter(b, 'wp', hp_cutoff_period)
    b = hp_filter(b, 'v', hp_cutoff_period)
    b = hp_filter(b, 'PP', hp_cutoff_period)
    b = hp_filter(b, 'q', hp_cutoff_period)
    # Calculate wave fluxes
    hp_vwf = b.hp_PP * b.hp_wp
    hp_vwf_nh = b.hp_q * b.hp_wp
    hp_hwf = b.hp_PP * b.hp_v
    hp_hwf_nh = b.hp_q * b.hp_v
    hp_fluxes = xr.merge([{'vwf': hp_vwf, 'vwf_nh': hp_vwf_nh, 'hwf': hp_hwf, 'hwf_nh': hp_hwf_nh}])
    print('saving high-pass filter-based model wave fluxes')
    nsl.io.close_nc(cfg.model.output.hp_fluxes)
    hp_fluxes.to_netcdf(cfg.model.output.hp_fluxes)
    # Integrate vertical wave flux
    intmin = cfg.parameters.model.integration_horizontal_min
    intmax = cfg.parameters.model.integration_horizontal_max
    hp_vwfi = integrate_vertical_iw_flux(hp_vwf, b, intmin, intmax)
    hp_vwfi_nh = integrate_vertical_iw_flux(hp_vwf_nh, b, intmin, intmax)
    hp_fluxes_int = xr.merge([{'vwf_int': hp_vwfi, 'vwf_int_nh': hp_vwfi_nh}])
    print('saving high-pass filter-based integrated model vertical wave fluxes')
    nsl.io.close_nc(cfg.model.output.hp_vwf_integrated)
    hp_fluxes_int.to_netcdf(cfg.model.output.hp_vwf_integrated)

