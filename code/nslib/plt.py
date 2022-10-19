#!/usr/bin/env python
# coding: utf-8
"""
Routines for generating various plots.
"""

import gvpy as gv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import cumtrapz

import nslib as nsl


def PlotKEold(a):
    ni = ["t12", "t14"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for axi in ax:
        axi = gv.plot.axstyle(axi)
    for i, k in enumerate(ni):
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        ty.KE.plot(ax=ax[i], cmap="RdYlBu_r", vmax=150)
        ax[i].invert_yaxis()
        ax[i].fill_between(ty.dist, ty.topo, 10000, color="0.3")
        ax[i].set(ylim=(5300, 4000), xlim=(0, 32))


def PlotKE(a):
    """Plot kinetic energy.

    Parameters
    ----------
    a : towyo data
    """
    cmap = "BuPu"
    # cmap = "PuRd"
    ni = ["t12", "t14"]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for axi in ax:
        axi = gv.plot.axstyle(axi)
    for i, k in enumerate(ni):
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        ty.KE.plot(
            ax=ax[i],
            cmap=cmap,
            extend="max",
            vmin=0,
            vmax=150,
            cbar_kwargs={"aspect": 70},
        )
        _plot_interface(ty, ax=ax[i])
        ax[i].invert_yaxis()
        ax[i].fill_between(ty.dist, ty.topo, 10000, color="0.3")
        ax[i].set(ylim=(5250, 4100), xlim=(0, 32))
        gv.plot.annotate_corner(
            ty.attrs["name"],
            ax[i],
            addy=-0.03,
            addx=0.02,
            fs=13,
            col="w",
            quadrant=2,
        )
    ax[1].set(ylabel="")
    ax[0].set(ylabel="depth [m]")


def PlotAPE(a):
    """Plot available potential energy.

    Parameters
    ----------
    a : towyo data
    """
    cmap = "BuPu"
    # cmap = "PuRd"
    ni = ["t12", "t14"]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    for axi in ax:
        axi = gv.plot.axstyle(axi)
    for i, k in enumerate(ni):
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        ty.APE.plot(
            ax=ax[i],
            cmap=cmap,
            extend="max",
            vmin=0,
            vmax=150,
            cbar_kwargs={"aspect": 70},
        )
        _plot_interface(ty, ax=ax[i])
        ax[i].invert_yaxis()
        ax[i].fill_between(ty.dist, ty.topo, 10000, color="0.3")
        ax[i].set(ylim=(5250, 4100), xlim=(0, 32))
        gv.plot.annotate_corner(
            ty.attrs["name"],
            ax[i],
            addy=-0.03,
            addx=0.02,
            fs=13,
            col="w",
            quadrant=2,
        )
    ax[1].set(ylabel="")
    ax[0].set(ylabel="depth [m]")


def PlotKEflux(a):
    ni = ["t12", "t14"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for axi in ax:
        axi = gv.plot.axstyle(axi)
    for i, k in enumerate(ni):
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        ty.KEflux.plot(ax=ax[i], cmap="RdBu_r", vmax=100)
        _plot_interface(ty, ax=ax[i])
        ax[i].invert_yaxis()
        ax[i].fill_between(ty.dist, ty.topo, 10000, color="0.3")
        ax[i].set(ylim=(5300, 4000), xlim=(0, 32))


def PlotAPEflux(a):
    ni = ["t12", "t14"]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    for axi in ax:
        axi = gv.plot.axstyle(axi)
    for i, k in enumerate(ni):
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        ty.APEflux.plot(ax=ax[i], cmap="RdBu_r", vmax=100)
        ty.gsw_sigma4.plot.contour(
            ax=ax[i], levels=[_InterfaceSG4], colors="k", linewidths=0.5
        )
        ax[i].invert_yaxis()
        ax[i].fill_between(ty.dist, ty.topo, 10000, color="0.3")
        ax[i].set(ylim=(5300, 4000), xlim=(0, 32))


def PlotVolumeFlux(a, fgs=(6, 4)):
    """Plot volume flux.

    Parameters
    ----------
    a : dict
        Towyo data.
    fgs : tuple, optional
        Figure size (width, height). Defaults to (6, 4).
    """
    fig, ax = gv.plot.quickfig(fgs=fgs)
    col = _linecolors(ax, option=0)
    for k, t in a.items():
        ax.plot(t.dist, t.vVI, linewidth=0.7, alpha=0.5)
    for k, t in a.items():
        t.vVI.rolling(x=20, center=True).mean().plot(
            ax=ax, x="dist", linewidth=1.5, label=t.attrs["name"]
        )
    ax.set(
        xlabel="dist [km]",
        ylabel=r"volume transport [m$^2$/s]",
        ylim=(-30, 200),
    )
    ax.axhline(color="k")
    ax.legend()
    gv.plot.png("towyo_volume_flux")


def PlotEnergeticsTerms(a, b, E, fluxes, cfg):
    """Plot energetics terms.

    Parameters
    ----------
    a : dict
        Towyo data.
    b : xarray.Dataset
        Model snapshot
    E : xarray.Dataset
        Model energy snapshot
    fluxes : xarray.Dataset
        Model energy fluxes based on localized spatial means. See
        :fun:`nslib.model_wave_fluxes.small_scale_flux_calculations` for how
        these are calculated.
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`

    Notes
    -----
    This function plots energetics fields:

    - APE
    - KE
    - APE flux
    - KE flux
    - horizontal pressure work
    - vertical pressure work
    - vertical IW flux

    for both observations and model.

    See Also
    --------
    PlotEnergeticsTermsIntegrated
    """
    # extract snapshot from model small scale fluxes
    ti = cfg.parameters.model.snapshot_ind
    flux = fluxes.isel(time=ti)
    # set up axes
    hs = 0.1
    nrows = 7
    fig = plt.figure(constrained_layout=False, figsize=(12, 19))
    gs0 = fig.add_gridspec(nrows=nrows, ncols=1, left=0.05, right=0.3, hspace=hs)
    gsc0 = fig.add_gridspec(nrows=nrows, ncols=1, left=0.31, right=0.315, hspace=hs)
    gs1 = fig.add_gridspec(nrows=nrows, ncols=1, left=0.33, right=0.58, hspace=hs)
    gsc1 = fig.add_gridspec(
        nrows=nrows, ncols=1, left=0.59 - 0.02, right=0.595 - 0.02, hspace=hs
    )
    gs2 = fig.add_gridspec(
        nrows=nrows, ncols=1, left=0.67 - 0.02, right=0.92 - 0.02, hspace=hs
    )
    gsc2 = fig.add_gridspec(
        nrows=nrows, ncols=1, left=0.93 - 0.02, right=0.935 - 0.02, hspace=hs
    )
    ax = []
    cax = []
    for i in range(nrows):
        ax.append(fig.add_subplot(gs0[i]))
        cax.append(fig.add_subplot(gsc0[i]))
        ax.append(fig.add_subplot(gs1[i]))
        cax.append(fig.add_subplot(gsc1[i]))
        ax.append(fig.add_subplot(gs2[i]))
        cax.append(fig.add_subplot(gsc2[i]))
    ax = np.array(ax).reshape((nrows, 3))
    cax = np.array(cax).reshape((nrows, 3))
    for axi in ax.flatten():
        axi = gv.plot.axstyle(axi, ticks="in", ticklength=2)
    for caxi in cax[:, 0].flatten():
        caxi.set_axis_off()

    # APE
    _plot_energetics_fields(
        a,
        ax[0, :2],
        cax[0, :2],
        "APEs",
        cmap="Blues",
        vmax=120,
        name=r"available potential energy $E^\prime_P$",
        cbar_kwargs=dict(label=r"$E^\prime_P$ [J/m$^3$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[0, 2],
        cax[0, 2],
        field="Epp_sorted_rho",
        name="model APE",
        vmin=0,
        vmax=120,
        cmap="Blues",
        cbar_kwargs=dict(label=r"$E^\prime_P$ [J/m$^3$]"),
    )
    # KE
    _plot_energetics_fields(
        a,
        ax[1, :2],
        cax[1, :2],
        "KE",
        cmap="Blues",
        vmax=120,
        name=r"kinetic energy $E^\prime_K$",
        cbar_kwargs=dict(label=r"$E^\prime_K$ [J/m$^3$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[1, 2],
        cax[1, 2],
        field="Ekp",
        name="model KE",
        vmin=0,
        vmax=120,
        cmap="Blues",
        cbar_kwargs=dict(label=r"$E^\prime_K$ [J/m$^3$]"),
    )
    # APE flux
    _plot_energetics_fields(
        a,
        ax[2, :2],
        cax[2, :2],
        "APEsflux",
        cmap="Purples",
        vmax=15,
        name=r"APE flux $v^\prime E^\prime_P$",
        cbar_kwargs=dict(label=r"$\mathbf{u}\,E^\prime_P$ [W/m$^2$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[2, 2],
        cax[2, 2],
        field="Eppflux",
        name="model APE flux",
        vmin=0,
        vmax=15,
        cmap="Purples",
        cbar_kwargs=dict(label=r"$v^\prime\,E^\prime_P$ [W/m$^2$]"),
    )
    # KE flux
    _plot_energetics_fields(
        a,
        ax[3, :2],
        cax[3, :2],
        "KEflux",
        cmap="Purples",
        vmax=30,
        name=r"KE flux $v^\prime E^\prime_K$",
        cbar_kwargs=dict(label=r"$\mathbf{u}\,E^\prime_K$ [W/m$^2$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[3, 2],
        cax[3, 2],
        field="Ekpflux",
        name="model KE flux",
        vmin=0,
        vmax=30,
        cmap="Purples",
        cbar_kwargs=dict(label=r"$v^\prime\,E^\prime_K$ [W/m$^2$]"),
    )
    # horizontal pressure work
    _plot_energetics_fields(
        a,
        ax[4, :2],
        cax[4, :2],
        "PressureWorkHorizSorted",
        cmap="PuOr_r",
        vmax=30,
        name=r"horizontal pressure work $v' p'$",
        vmin=-30,
        cbar_kwargs=dict(label=r"$v'\,p'$ [W/m$^2$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[4, 2],
        cax[4, 2],
        field="IWEnergyFluxHoriz_sorted_rho",
        name="model IW flux",
        vmin=-30,
        vmax=30,
        cmap="PuOr_r",
        cbar_kwargs=dict(label=r"$v'\,p'$ [W/m$^2$]"),
    )
    # vertical pressure work
    _plot_energetics_fields(
        a,
        ax[5, :2],
        cax[5, :2],
        "PressureWorkVertSorted",
        cmap="PuOr_r",
        vmax=2,
        name=r"vertical pressure work $w' p'$",
        vmin=-2,
        cbar_kwargs=dict(label=r"$w'\,p'$ [W/m$^2$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        E,
        ax[5, 2],
        cax[5, 2],
        field="IWEnergyFluxVert_sorted_rho",
        name="model IW flux",
        vmin=-2,
        vmax=2,
        cmap="PuOr_r",
        cbar_kwargs=dict(label=r"$w'\,p'$ [W/m$^2$]"),
    )
    # vertical IW flux
    _plot_energetics_fields(
        a,
        ax[6, :2],
        cax[6, :2],
        "wp_pp",
        cmap="PuOr_r",
        vmax=1,
        name=r"vertical IW flux $w'' p''$",
        vmin=-1,
        cbar_kwargs=dict(label=r"$w''\,p''$ [W/m$^2$]"),
    )
    _plot_energetics_fields_model(
        a,
        b,
        flux,
        ax[6, 2],
        cax[6, 2],
        field="vwf",
        name="model IW flux",
        vmin=-1,
        vmax=1,
        cmap="PuOr_r",
        cbar_kwargs=dict(label=r"$w''\,p''$ [W/m$^2$]"),
    )

    for axi in ax.flatten():
        axi.set(ylabel="")
        axi.set(yticks=np.arange(4250, 5250, 250))
    for axi in ax[:-1, :].flatten():
        axi.set(xlabel="")
        axi.set_xticklabels([])
    for axi in ax[:, 1:].flatten():
        axi.set_yticklabels([])
    ax[0, 0].set_title("towyo 2012", fontdict={"fontweight": "bold"})
    ax[0, 1].set_title("towyo 2014", fontdict={"fontweight": "bold"})
    ax[0, 2].set_title("model", fontdict={"fontweight": "bold"})
    ax[3, 0].set(ylabel="depth [m]")
    for axi in ax[-1, :]:
        axi.set(xlabel="")
    ax[-1, 1].set(xlabel="distance [km]")

    # add subplot labels
    gv.plot.subplotlabel(ax, color="k", fs=10, fw="bold", bg="w", bga=1)


def _plot_energetics_fields(
    a, ax, cax, field, cmap, vmax, name, vmin=0, cbar_kwargs={}
):
    ni = ["t12", "t14"]
    extend = "max" if vmin == 0 else "both"
    # KE
    for i, k in enumerate(ni):
        axx = ax[i]
        ty = a[k]
        ty = ty.swap_dims({"x": "dist"})
        h = ty[field].plot(
            ax=axx,
            cmap=cmap,
            extend="max",
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            rasterized=True,
        )
        _plot_interface(ty, ax=axx)
        axx.invert_yaxis()
        axx.fill_between(ty.dist, ty.topo, 10000, color="0.3")
        axx.set(ylim=(5250, 4100), xlim=(0, 32))
        if i == 0:
            gv.plot.annotate_corner(
                name,
                axx,
                addy=-0.06,
                addx=-0.01,
                fs=10,
                col="w",
                quadrant=2,
                fw="normal",
            )
        if i == 1:
            plt.colorbar(h, cax=cax[-1], **cbar_kwargs)


def _plot_energetics_fields_model(
    a, b, E, ax, cax, field, cmap, vmax, name, vmin=0, cbar_kwargs={}
):
    extend = "max" if vmin == 0 else "both"
    ty = a["t12"]
    axx = ax
    h = E[field].plot(
        ax=axx,
        cmap=cmap,
        extend="max",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        rasterized=True,
    )
    plt.colorbar(h, cax=cax, **cbar_kwargs)
    # b.th.plot.contour(ax=axx, levels=[0.9], colors="k", linewidths=0.5)
    b.th.plot.contour(ax=axx, levels=[0.8, 0.9], colors="k", linewidths=0.5)
    axx.invert_yaxis()
    axx.fill_between(ty.dist, ty.topo, 10000, color="0.3")
    axx.set(ylim=(5250, 4100), xlim=(0, 32), title="")
    return h


def PlotEnergeticsTermsIntegrated(a, b, E, model_int_limit_isot=0.9):
    """Plot energetics terms (integrated along x or z).

    Parameters
    ----------
    a : dict
        Towyo data.
    b : xarray.Dataset
        Model snapshot
    E : xarray.Dataset
        Model energy snapshot
    model_int_limit_isot : float or list
        Model upper integration limit isotherm. Can either be a float or a list
        of two floats for two isotherms.

    Notes
    -----
    This function plots depth-integrated energetics fields from both
    observations and a model snapshot:

    - volume flux
    - APE flux
    - KE flux
    - turbulent dissipation
    - bottom drag dissipation

    This is the figure:

    .. figure:: ../fig/energetics_fields_integrated.png
        :figwidth: 400px
        :align: center

        Depth-integrated terms of the energy equation for observations and model.


    See Also
    --------
    PlotEnergeticsTerms

    """
    # Parameters for the windowed mean, currently corresponding to about 2 km.
    towyo_smoothing_factor = 10
    model_smoothing_factor = 100
    if type(model_int_limit_isot) == float:
        model_int_limit_isot = [model_int_limit_isot]
    model_colors = ["C6", "#F06292"]
    model_line_styles = ["-", "--"]
    fig = plt.figure(figsize=(11, 17), constrained_layout=False)
    gs0 = fig.add_gridspec(nrows=6, ncols=1, left=0.05, right=0.55)
    # gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.65, right=0.9)
    ax = []
    for i in range(6):
        ax.append(fig.add_subplot(gs0[i]))
    # for i in range(2):
    #     ax.append(fig.add_subplot(gs1[i]))
    ax = np.array(ax)
    ax = ax.flatten()
    for axi in ax:
        gv.plot.axstyle(axi, fontsize=12, ticks="in")

    optsraw = dict(linewidth=0.7, alpha=0.5)
    optssmooth = dict(linewidth=1.5, alpha=1)

    # volume flux obs
    axi = ax[0]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    for k, t in a.items():
        axi.plot(t.dist, t.vVI, **optsraw)
    for k, t in a.items():
        t.vVI.rolling(x=towyo_smoothing_factor, center=True).mean().plot(
            ax=axi,
            x="dist",
            label=t.attrs["name"],
            **optssmooth,
        )
    xlims = axi.get_xlim()
    # model snapshot
    for color, limit, linestyle in zip(
        model_colors, model_int_limit_isot, model_line_styles
    ):
        mvI = b.v.where(((b.th < limit) & np.isfinite(b.v)), 0).integrate(coord="depth")
        mvI.plot(ax=axi, color=color, linestyle=linestyle, **optsraw)
        mvI.rolling(dist=model_smoothing_factor).mean().plot(
            ax=axi, color=color, linestyle=linestyle, **optssmooth
        )
    axi.set(
        xlabel="dist [km]",
        ylabel=r"$\int v^\prime\,\mathrm{dz}\ $ [m$^2$/s]",
        ylim=(-30, 180),
        xlim=xlims,
    )
    # axi.axhline(color="k", linewidth=1)
    gv.plot.annotate_corner("a) volume flux", ax=axi, addx=-0.02)

    # APE flux obs
    # Add an offset to account for slightly offset downstream reference density
    # when using the sorted density field as reference. This does not affect
    # the budget, only avoids negative flux downstream.
    ape_offset = 1
    axi = ax[1]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    for k, ty in a.items():
        (ty["APEsfluxVI"] / 1e3 + ape_offset).swap_dims({"x": "dist"}).plot(
            ax=axi, **optsraw
        )
    h = []
    for k, ty in a.items():
        htmp = (
            (ty["APEsfluxVI"] / 1e3 + ape_offset)
            .rolling(x=towyo_smoothing_factor, center=True)
            .mean()
            .swap_dims({"x": "dist"})
            .plot(ax=axi, **optssmooth, label="20" + k[1:])
        )
        h.append(htmp[0])
    # model
    mAPEflux = b.v * E.Epp_sorted_rho / 1e3  # convert to kW/m
    for color, limit, linestyle in zip(
        model_colors, model_int_limit_isot, model_line_styles
    ):
        mAPEfluxI = (
            mAPEflux.where(((b.th < limit) & np.isfinite(b.v)), 0).integrate(
                coord="depth"
            )
            + ape_offset
        )
        mAPEfluxI.plot(ax=axi, linestyle=linestyle, color=color, **optsraw)
        htmp = (
            mAPEfluxI.rolling(dist=model_smoothing_factor, center=True)
            .mean()
            .plot(
                ax=axi,
                label=f"model {limit}$^{{\circ}}$C",
                color=color,
                linestyle=linestyle,
                **optssmooth,
            )
        )
        h.append(htmp[0])
    axi.legend(handles=h, loc="upper right")
    axi.set(ylabel=r"$\int v^\prime E^\prime_P\, \mathrm{dz}\ $ [kW/m]", xlim=xlims)
    ax[1].set_ylim(bottom=-2, top=7)
    gv.plot.annotate_corner("b) APE flux", ax=axi, quadrant=1, addx=-0.02)

    # KE flux obs
    axi = ax[2]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    for k, ty in a.items():
        (ty["KEfluxVI"] / 1e3).swap_dims({"x": "dist"}).plot(ax=axi, **optsraw)
    for k, ty in a.items():
        (ty["KEfluxVI"] / 1e3).rolling(
            x=towyo_smoothing_factor, center=True
        ).mean().swap_dims({"x": "dist"}).plot(ax=axi, **optssmooth)
    # model
    mKEflux = b.v * E.Ekp / 1e3  # convert to kW/m
    for color, limit, linestyle in zip(
        model_colors, model_int_limit_isot, model_line_styles
    ):
        mKEfluxI = mKEflux.where(((b.th < limit) & np.isfinite(b.v)), 0).integrate(
            coord="depth"
        )
        mKEfluxI.plot(ax=axi, color=color, linestyle=linestyle, **optsraw)
        mKEfluxI.where(~np.isnan(mKEfluxI), drop=True).rolling(
            dist=model_smoothing_factor, center=True
        ).mean().plot(ax=axi, color=color, linestyle=linestyle, **optssmooth)
    axi.set(ylabel=r"$\int v^\prime E^\prime_K\, \mathrm{dz}\ $ [kW/m]", xlim=xlims)
    ax[2].set_ylim(bottom=-2)
    gv.plot.annotate_corner("c) KE flux", ax=axi, addx=-0.02)

    # Horizontal Pressure Work
    axi = ax[3]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    for k, ty in a.items():
        (ty["PressureWorkHorizSortedVI"] / 1e3).swap_dims({"x": "dist"}).plot(
            ax=axi, **optsraw
        )
    for k, ty in a.items():
        (ty["PressureWorkHorizSortedVI"] / 1e3).rolling(
            x=towyo_smoothing_factor, center=True
        ).mean().swap_dims({"x": "dist"}).plot(ax=axi, **optssmooth)
    # model
    mPressureWorkHoriz = E.IWEnergyFluxHoriz_sorted_rho / 1e3
    for color, limit, linestyle in zip(
        model_colors, model_int_limit_isot, model_line_styles
    ):
        mPressureWorkHorizI = mPressureWorkHoriz.where(
            ((b.th < limit) & np.isfinite(b.v)), 0
        ).integrate(coord="depth")
        mPressureWorkHorizI.plot(ax=axi, color=color, linestyle=linestyle, **optsraw)
        mPressureWorkHorizI.where(~np.isnan(mPressureWorkHorizI), drop=True).rolling(
            dist=model_smoothing_factor, center=True
        ).mean().plot(ax=axi, color=color, linestyle=linestyle, **optssmooth)
    axi.set(ylabel=r"$\int v^\prime p^\prime\, \mathrm{dz}\ $ [kW/m]", xlim=xlims)
    axi.set_ylim(bottom=-14, top=14)
    gv.plot.annotate_corner("d) horizontal pressure work", ax=axi, addx=-0.02)

    # epsilon obs
    axi = ax[4]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    [
        axi.plot(
            ty.dist,
            ty.LowerLayerEpsIntegrated / 1e3,
            label="20" + k[1:],
            **optssmooth,
        )
        for k, ty in a.items()
    ]
    # model
    mrhoeps = b.eps.where(~np.isnan(b.eps), 0) * b.rho.where(~np.isnan(b.rho), 0)
    for color, limit, linestyle in zip(
        model_colors, model_int_limit_isot, model_line_styles
    ):
        mrhoepsI = mrhoeps.where(((b.th < limit)), 0).integrate(coord="depth")

        mrhoepsIc = cumtrapz(mrhoepsI, b.dist * 1e3, initial=0)
        axi.plot(
            b.dist,
            mrhoepsIc / 1e3,
            color=color,
            linestyle=linestyle,
            **optssmooth,
        )
    axi.set(
        ylabel=r"$\rho \int \epsilon\, \mathrm{dz}\, \mathrm{dy}\ $ [kW/m]",
        xlim=xlims,
    )
    axi.set(xlabel="Distance [km]")
    axi.set_ylim(bottom=0, top=2.3)
    gv.plot.annotate_corner(
        "e) turbulent dissipation (cumulative integral)",
        ax=axi,
        quadrant=1,
        addx=-0.02,
    )

    # bottom drag dissipation obs
    axi = ax[5]
    axi.spines["bottom"].set_position(("data", 0))
    col = _linecolors(axi, option=0)
    [
        axi.plot(
            ty.dist,
            ty.BottomDragDissipationIntegrated / 1e3,
            label="20" + k[1:],
            **optssmooth,
        )
        for k, ty in a.items()
    ]
    # model
    mbd = E.BottomDrag.where(~np.isnan(E.BottomDrag), 0)
    mbdIc = cumtrapz(mbd, b.dist * 1e3, initial=0)
    axi.plot(b.dist, mbdIc / 1e3, color="C6", **optssmooth)
    # also the version calculated from velocities 40m above bottom
    mbda = E.BottomDragAbove.where(~np.isnan(E.BottomDragAbove), 0)
    mbdIca = cumtrapz(mbda, b.dist * 1e3, initial=0)
    axi.plot(b.dist, mbdIca / 1e3, color="C7", **optssmooth)
    i = np.absolute(b.dist - 30).argmin().data
    axi.annotate(
        "from vel at bottom",
        xy=(b.dist[i], mbdIc[i] / 1e3),
        xytext=(0, 20),
        xycoords="data",
        textcoords="offset pixels",
        color="C6",
        ha="center",
    )
    axi.annotate(
        "from vel 40m\n above bottom",
        xy=(b.dist[i], mbdIca[i] / 1e3),
        xytext=(0, 20),
        xycoords="data",
        textcoords="offset pixels",
        color="C7",
        ha="center",
    )

    axi.set(
        ylabel=r"$\int D^\prime \, \mathrm{dy}\ $ [kW/m]",
        xlim=xlims,
    )
    axi.set(xlabel="Distance [km]")
    axi.set_ylim(bottom=0, top=4)
    gv.plot.annotate_corner(
        "f) bottom drag dissipation (cumulative integral)",
        ax=axi,
        quadrant=1,
        addx=-0.02,
    )

    # # IW fluxes
    # axi = ax[4]
    # col = _linecolors(axi, option=0)
    # for k, ty in a.items():
    #     (ty['wp_pp_mean_sg4class']).plot(y='sg4bins', yincrease=False, ax=axi, **optssmooth)
    # axi.set(ylabel=r'$\sigma_4\ $ [kg/m$^3$ - 1000]', xlabel=r'$w^\prime p^\prime$ [W/m$^2$]')

    for axi in ax[:5]:
        axi.set_xticklabels("")
        axi.set(title="")

    for axi in ax[:5]:
        axi.set(xlabel="")


def PlotVerticalEnergyFlux(cfg, a):
    """Plot vertical wave energy flux (integrated along x).

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    a : dict
        Towyo data.

    See Also
    --------
    PlotEnergeticsTerms, PlotEnergeticsTermsIntegrated,
    :fun:`nslib.model_wave_fluxes.small_scale_flux_calculations`
    """
    fontsize = 12
    fig, ax2, ax = gv.plot.newfigyy(width=5, height=6)
    for axi in [ax, ax2]:
        ax = gv.plot.axstyle(ax, fontsize=fontsize)
    col = _linecolors(ax, option=0)
    h1 = []
    # towyo integrated fluxes:
    for k, t in a.items():
        htmp = gv.plot.vstep(
            t.wp_pp_along_isopycnal / 1e3, t.sg4bins, label="20" + k[1:]
        )
        h1.append(htmp[0])
    # model integrated fluxes:
    # load horizontally integrated model wave energy flux time series
    vwfi = xr.open_dataarray(cfg.model.small_scale_vwf_integrated)
    vwfim = vwfi.mean(dim="time")
    h = gv.plot.vstep(vwfim / 1e3, vwfim.isot, ax=ax2, color="C6", label="model")
    h1.append(
        h[0],
    )
    # model wave energy flux time series based on high-pass filter
    hpvwfi = xr.open_dataset(cfg.model.hp_vwf_integrated)
    hpvwfim = hpvwfi.vwf_int.mean(dim="time")
    h = gv.plot.vstep(
        hpvwfim / 1e3, hpvwfim.isot, ax=ax2, color="#F57C00", label="model hp"
    )
    h1.append(
        h[0],
    )
    ax.set(
        ylim=(45.99, 45.86),
        xlim=(-0.5, 2.5),
        # xlabel=r"$\int w^\prime p^\prime\ dy$ [W/m]",
        ylabel=r"$\sigma_4$ [kg/m$^3$ - 1000]",
    )
    ax.legend(h1, ["towyo 2012", "towyo 2014", "model", "model hp"], fontsize=11)
    ax.hlines(45.94, -1e5, 1e5, color="k", linestyle="--", zorder=1, alpha=0.5)
    ax2.hlines(0.9, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
    ax2.hlines(0.8, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
    ax2.annotate(
        "interface",
        xy=(1.5, 0.9),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )
    ax2.annotate(
        "model 0.8$^{\circ}$C interface",
        xy=(0.7, 0.8),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )
    # ax.spines["left"].set_position(("data", 0))
    ax.set(ylim=(45.99, 45.92))
    ax2.set(
        ylim=(0.65, 1.0),
        ylabel=r"$\theta$ [$^{\circ}$C]",
        xlabel=r"$\int w'' p''\ dy$ [kW/m]",
    )
    ax2.spines["right"].set_visible(True)
    almost_black = "#262626"
    spine = "right"
    ax2.spines[spine].set_linewidth(0.5)
    ax2.spines[spine].set_color(almost_black)
    ax2.spines[spine].set_position(("outward", 5))
    ax2.yaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_size(fontsize)
    ax2.yaxis.offsetText.set_fontsize(fontsize)
    ax2.grid(False)
    ax2.vlines(0, -10, 10, color="0.5", linewidth=1)


def PlotVerticalEnergyFluxWithPressureWork(cfg, wppp_int, a):
    """Plot vertical wave energy flux (integrated along x).

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    a : dict
        Towyo data.

    See Also
    --------
    PlotEnergeticsTerms, PlotEnergeticsTermsIntegrated,
    :fun:`nslib.model_wave_fluxes.small_scale_flux_calculations`
    """
    fontsize = 12
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True, sharey=True
    )
    ax1, ax2 = ax
    ax1a = ax1.twinx()
    ax2a = ax2.twinx()

    for axi in [ax1, ax2, ax1a, ax2a]:
        gv.plot.axstyle(axi)

    for axi in [ax1, ax2]:
        axi.set(ylim=(0.65, 1.0))

    ax1.set(
        ylabel=r"$\theta$ [$^{\circ}$C]",
        xlabel=r"$\int w' p'\ dy$ [kW/m]",
        xlim=(-4.5, 4.5),
    )

    ax2.set(
        ylabel="",
        xlabel=r"$\int w'' p''\ dy$ [kW/m]",
        xlim=(-0.2, 2.2),
    )

    ax1a.yaxis.set_ticklabels([])

    ax2a.set(
        ylabel=r"$\sigma_4$ [kg/m$^3$ - 1000]",
    )

    for axi in [ax1a, ax2a]:
        axi.spines["right"].set_visible(True)
        almost_black = "#262626"
        spine = "right"
        axi.spines[spine].set_linewidth(0.5)
        axi.spines[spine].set_color(almost_black)
        axi.spines[spine].set_position(("outward", 5))
        axi.yaxis.label.set_color(almost_black)
        axi.yaxis.label.set_size(fontsize)
        axi.yaxis.offsetText.set_fontsize(fontsize)
        axi.grid(False)
        # axi.vlines(0, -10, 10, color="0.5", linewidth=1)
        axi.set(ylim=(45.99, 45.92))

    ax2.spines["left"].set_visible(False)
    # ax2a.spines["left"].set_visible(False)
    ax1a.spines["right"].set_visible(False)

    gv.plot.annotate_corner("a) vertical pressure work", ax=ax1, addx=-0.02, addy=0.12)
    gv.plot.annotate_corner("b) vertical internal waves", ax=ax2, addx=-0.02, addy=0.12)

    for axi in [ax1, ax2]:
        axi.vlines(0, -10, 10, color=almost_black, linewidth=1)

    for axi in [ax1, ax2]:
        axi.hlines(0.9, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
        axi.hlines(0.8, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
    ax1.annotate(
        "interface",
        xy=(-3.8, 0.9),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )
    ax1.annotate(
        "model 0.8$^{\circ}$C interface",
        xy=(-4.3, 0.8),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )

    col = _linecolors(ax1a, option=0)
    col = _linecolors(ax2a, option=0)

    h1 = []
    # towyo integrated fluxes:
    for k, t in a.items():
        htmp = gv.plot.vstep(
            t.wp_pp_along_isopycnal / 1e3,
            t.sg4bins,
            label="20" + k[1:],
            ax=ax2a,
        )
        h1.append(htmp[0])
    # model integrated fluxes:
    # load horizontally integrated model wave energy flux time series
    vwfi = xr.open_dataarray(cfg.model.small_scale_vwf_integrated)
    vwfim = vwfi.mean(dim="time")
    h = gv.plot.vstep(vwfim / 1e3, vwfim.isot, ax=ax2, color="C6", label="model")
    h1.append(h[0])
    # model wave energy flux time series based on high-pass filter
    hpvwfi = xr.open_dataset(cfg.model.hp_vwf_integrated)
    hpvwfim = hpvwfi.vwf_int.mean(dim="time")
    h = gv.plot.vstep(
        hpvwfim / 1e3, hpvwfim.isot, ax=ax2, color="#F57C00", label="model hp"
    )
    h1.append(h[0])

    h0 = []
    # towyo pressure work
    for k, t in a.items():
        htmp = gv.plot.vstep(
            t.PressureWorkVertSorted_along_isopycnal / 1e3,
            t.sg4bins,
            label="20" + k[1:],
            linestyle="-",
            ax=ax1a,
        )
        h0.append(htmp[0])
    # model pressure work
    h = gv.plot.vstep(
        wppp_int / 1e3,
        wppp_int.isot,
        ax=ax1,
        color="C6",
        linestyle="-",
        label=r"model $w'p'$",
    )
    h0.append(h[0])

    ax1.legend(
        h0,
        ["towyo 2012", "towyo 2014", "model"],
        fontsize=11,
        loc=2,
    )
    ax2.legend(h1, ["towyo 2012", "towyo 2014", "model", "model hp"], fontsize=11)


def PlotEnergeticsLayerBudgetResultsHorizontal(cfg):
    """Plot energy budget results with horizontal bars

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    """
    # Read model energy budget results
    B = xr.open_dataset(cfg.model.energy_budget_results_layer)
    B2 = xr.open_dataset(cfg.model.energy_budget_results_box)
    plot_terms = [
        "Unsteadiness",
        "EppFluxDivergence",
        "EkpFluxDivergence",
        "IWFluxVertDivergence_hp",
        "IWFluxHorizDivergence_hp",
        "Dissipation",
        "BottomDrag",
    ]
    # model terms
    means = []
    stds = []
    for var in plot_terms:
        means.append(B[var].mean(dim="T").data)
        stds.append(B[var].std(dim="T").data)
    # print layer results
    print("-----")
    print("budget results for layer [kW/m]:")
    for name, data in zip(plot_terms, means):
        print(name, f"{data/1e3:1.2f}")
    print("-----")
    means = np.array(means)
    stds = np.array(stds)

    fig, ax = gv.plot.quickfig(fgs=(5.5, 7))
    col = _linecolors(ax, option=0)
    x = np.arange(len(plot_terms))
    width = 0.15
    h = []
    tmp = ax.barh(x, means / 1e3, width, color="C6", label="model layer")
    h.append(tmp[0])
    for xi, mi, si in zip(x, means / 1e3, stds / 1e3):
        ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
    addwidth = 1
    if B2 is not None:
        means = []
        stds = []
        for var in plot_terms:
            means.append(B2[var].mean(dim="T").data)
            stds.append(B2[var].std(dim="T").data)
        means = np.array(means)
        stds = np.array(stds)
        # print box results
        print("-----")
        print("budget results for box [kW/m]:")
        for name, data in zip(plot_terms, means):
            print(name, f"{data/1e3:1.2f}")
        print("-----")
        tmp = ax.barh(
            x + addwidth * width,
            means / 1e3,
            width,
            color="#F06292",
            label="model box",
        )
        h.append(tmp[0])
        for xi, mi, si in zip(x + addwidth * width, means / 1e3, stds / 1e3):
            ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
        addwidth += 1
    addwidth += 0.5

    # towyo energy budget results
    tyb = xr.open_dataset(cfg.obs.energy_budget)
    for year, ty in tyb.groupby("year"):
        tyterms = np.zeros_like(means)
        tyterms[1] = ty.apeflux_div
        tyterms[2] = ty.keflux_div
        tyterms[3] = ty.vertical_wave_flux
        tyterms[5] = ty.dissipation
        tyterms[6] = ty.bottom_dissipation
        tmp = ax.barh(x + addwidth * width, tyterms / 1e3, width, label=f"towyo {year}")
        h.append(tmp[0])
        ax.hlines(
            y=x[1] + addwidth * width,
            xmin=ty.apeflux_range[0] / 1e3,
            xmax=ty.apeflux_range[1] / 1e3,
            color="0.5",
            linewidth=1.5,
        )
        ax.hlines(
            y=x[2] + addwidth * width,
            xmin=ty.keflux_range[0] / 1e3,
            xmax=ty.keflux_range[1] / 1e3,
            color="0.1",
            linewidth=1.5,
            alpha=0.5,
        )
        addwidth += 1

    centerwidth = (addwidth - 1) * width / 2
    ax.spines["left"].set_visible(False)
    yticks = x[:-1] + width * (addwidth + 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set(ylim=[yticks[-1] + 1, yticks[0] - 1])
    ax.set(xlabel="Energy Flux [kW/m]")
    _add_bar_label_h(ax, "steadiness", 0, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "APE flux divergence", 1, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(ax, "KE flux divergence", 2, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(
        ax,
        "vertical internal wave\nflux divergence",
        3,
        centerwidth=centerwidth,
        lr="l",
    )
    _add_bar_label_h(
        ax,
        "horizontal internal wave\nflux divergence",
        4,
        centerwidth=centerwidth,
        offset=20,
        lr="l",
    )
    _add_bar_label_h(ax, "turbulent dissipation", 5, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "bottom drag", 6, centerwidth=centerwidth, lr="l")
    ax.grid(True)
    ax.legend(loc=4)
    gv.plot.xsym(ax=ax)


def PlotEnergeticsLayerBudgetResultsHorizontalNew(cfg, refrho_sorted=True):
    """Plot energy budget results with horizontal bars. This version includes
    pressure work terms v'p' and w'p'.

    Also, choose between the way we calculate a reference density profile.

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    refrho_sorted : bool
        Calculate background density based on a sorted background state. Defaults to True.
    """
    # Read model energy budget results and concatenate them for plotting.
    B = xr.open_dataset(cfg.model.energy_budget_results_layer)
    B2 = xr.open_dataset(cfg.model.energy_budget_results_box)
    if refrho_sorted:
        Epp = "EppSortedFluxDivergence"
        IWVert = "IWSortedFluxVertDivergence"
        IWHoriz = "IWSortedFluxHorizDivergence"
    else:
        Epp = "EppFluxDivergence"
        IWVert = "IWFluxVertDivergence"
        IWHoriz = "IWFluxHorizDivergence"
    plot_terms = [
        "Unsteadiness",
        Epp,
        "EkpFluxDivergence",
        IWVert,
        IWHoriz,
        "IWFluxVertDivergence_hp",
        "IWFluxHorizDivergence_hp",
        "Dissipation",
        "BottomDrag",
    ]
    means = []
    stds = []
    for var in plot_terms:
        means.append(B[var].mean(dim="T").data)
        stds.append(B[var].std(dim="T").data)
    means = np.array(means)
    stds = np.array(stds)
    # print layer results
    print("-----")
    print("budget results for layer [kW/m]:")
    for name, data in zip(plot_terms, means):
        print(name, f"{data/1e3:1.2f}")
    sum_terms = means[[1, 2, 3, 4, 7, 8]]
    print("terms for residuals:", sum_terms)
    residual = np.sum(means[[1, 2, 3, 4, 7, 8]])
    print(f"residual: {residual:1.2f}")
    print("-----")

    fig, ax = gv.plot.quickfig(fgs=(5.5, 7))
    col = _linecolors(ax, option=0)
    x = np.arange(len(plot_terms))
    width = 0.15
    h = []
    tmp = ax.barh(x, means / 1e3, width, color="C6", label="model layer")
    h.append(tmp[0])
    for xi, mi, si in zip(x, means / 1e3, stds / 1e3):
        ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
    addwidth = 1
    if B2 is not None:
        means = []
        stds = []
        for var in plot_terms:
            means.append(B2[var].mean(dim="T").data)
            stds.append(B2[var].std(dim="T").data)
        means = np.array(means)
        stds = np.array(stds)
        # print box results
        print("-----")
        print("budget results for box [kW/m]:")
        for name, data in zip(plot_terms, means):
            print(name, f"{data/1e3:1.2f}")
        print("-----")
        tmp = ax.barh(
            x + addwidth * width,
            means / 1e3,
            width,
            color="#F06292",
            label="model box",
        )
        h.append(tmp[0])
        for xi, mi, si in zip(x + addwidth * width, means / 1e3, stds / 1e3):
            ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
        addwidth += 1
    addwidth += 0.5

    # towyo energy budget results
    tyb = xr.open_dataset(cfg.obs.energy_budget)
    for year, ty in tyb.groupby("year"):
        tyterms = np.zeros_like(means)
        if refrho_sorted:
            tyterms[1] = ty.apesflux_div
            tyterms[2] = ty.keflux_div
            tyterms[3] = ty.vertical_pressure_work_sorted
            tyterms[4] = ty.horizontal_pressure_work_sorted
            tyterms[5] = ty.vertical_wave_flux
            tyterms[7] = ty.dissipation
            tyterms[8] = ty.bottom_dissipation
        else:
            tyterms[1] = ty.apeflux_div
            tyterms[2] = ty.keflux_div
            tyterms[3] = ty.vertical_pressure_work
            tyterms[4] = ty.horizontal_pressure_work
            tyterms[5] = ty.vertical_wave_flux
            tyterms[7] = ty.dissipation
            tyterms[8] = ty.bottom_dissipation
        tmp = ax.barh(x + addwidth * width, tyterms / 1e3, width, label=f"towyo {year}")
        h.append(tmp[0])
        # color = "0.2" if year==2012 else "0.5"
        # ax.hlines(
        #     y=x[1] + addwidth * width,
        #     xmin=ty.apesflux_range[0] / 1e3,
        #     xmax=ty.apesflux_range[1] / 1e3,
        #     color=color,
        #     linewidth=1.5,
        # )
        # ax.hlines(
        #     y=x[2] + addwidth * width,
        #     xmin=ty.keflux_range[0] / 1e3,
        #     xmax=ty.keflux_range[1] / 1e3,
        #     color=color,
        #     linewidth=1.5,
        #     alpha=0.5,
        # )
        # ax.hlines(
        #     y=x[4] + addwidth * width,
        #     xmin=ty.PressureWorkHoriz_range[0] / 1e3,
        #     xmax=ty.PressureWorkHoriz_range[1] / 1e3,
        #     color=color,
        #     linewidth=1.5,
        #     alpha=0.5,
        # )
        addwidth += 1

    centerwidth = (addwidth - 1) * width / 2
    ax.spines["left"].set_visible(False)
    yticks = x[:-1] + width * (addwidth + 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set(ylim=[yticks[-1] + 1, yticks[0] - 1])
    ax.set(xlabel="Energy Flux [kW/m]")
    _add_bar_label_h(ax, "steadiness", 0, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "APE flux divergence", 1, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(ax, "KE flux divergence", 2, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(
        ax,
        "vertical pressure work\ndivergence $w'p'$",
        3,
        centerwidth=centerwidth,
        lr="l",
    )
    _add_bar_label_h(
        ax,
        "horizontal pressure work\ndivergence $v'p'$",
        4,
        centerwidth=centerwidth,
        offset=20,
        lr="r",
    )
    _add_bar_label_h(
        ax,
        "vertical internal wave\nflux divergence $w''p''$",
        5,
        centerwidth=centerwidth,
        lr="l",
    )
    _add_bar_label_h(
        ax,
        "horizontal internal wave\nflux divergence $v''p''$",
        6,
        centerwidth=centerwidth,
        offset=20,
        lr="l",
    )
    _add_bar_label_h(ax, "turbulent dissipation", 7, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "bottom drag", 8, centerwidth=centerwidth, lr="l")
    ax.grid(True)
    ax.legend(loc=4)
    gv.plot.xsym(ax=ax)


def PlotEnergeticsLayerBudgetResultsHorizontalTwoIsotherms(cfg, refrho_sorted=True):
    """Plot energy budget results with horizontal bars. This version includes
    pressure work terms v'p' and w'p'. Also, this version does not show the box
    results but shows results for two different isotherms instead.

    Also, choose between the way we calculate a reference density profile.

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    refrho_sorted : bool
        Calculate background density based on a sorted background state. Defaults to True.
    """
    # Read model energy budget results and concatenate them for plotting.
    B = xr.open_dataset(cfg.model.energy_budget_results_layer)
    B2 = xr.open_dataset(cfg.model.energy_budget_results_layer_08)
    if refrho_sorted:
        Epp = "EppSortedFluxDivergence"
        IWVert = "IWSortedFluxVertDivergence"
        IWHoriz = "IWSortedFluxHorizDivergence"
    else:
        Epp = "EppFluxDivergence"
        IWVert = "IWFluxVertDivergence"
        IWHoriz = "IWFluxHorizDivergence"
    plot_terms = [
        "Unsteadiness",
        Epp,
        "EkpFluxDivergence",
        IWHoriz,
        IWVert,
        "IWFluxVertDivergence_hp",
        "Dissipation",
        "BottomDrag",
    ]
    means = []
    stds = []
    for var in plot_terms:
        means.append(B[var].mean(dim="T").data)
        stds.append(B[var].std(dim="T").data)
    means = np.array(means)
    stds = np.array(stds)
    # print layer results
    print("-----")
    print("budget results for layer [kW/m]:")
    for name, data, sigma in zip(plot_terms, means, stds):
        print(name, f"{data/1e3:1.2f} +- {sigma/1e3:1.2f}")
    sum_terms = means[[1, 2, 3, 4, 6, 7]]
    print("terms for residuals:", sum_terms)
    residual = np.sum(means[[1, 2, 3, 4, 6, 7]])
    print(f"residual: {residual/1e3:1.2f}")
    sigmas_squared = np.array(stds[[1, 2, 3, 4, 6, 7]]) ** 2
    combined_error = np.sqrt(np.sum(sigmas_squared))
    print(f"residual sigma: {combined_error/1e3:1.2f}")
    print("-----")

    fig, ax = gv.plot.quickfig(fgs=(5.5, 6))
    col = _linecolors(ax, option=0)
    x = np.arange(len(plot_terms))
    width = 0.15
    h = []
    tmp = ax.barh(x, means / 1e3, width, color="C6", label="model 0.9$^{\circ}$C")
    h.append(tmp[0])
    for xi, mi, si in zip(x, means / 1e3, stds / 1e3):
        ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
    addwidth = 1
    if B2 is not None:
        means = []
        stds = []
        for var in plot_terms:
            means.append(B2[var].mean(dim="T").data)
            stds.append(B2[var].std(dim="T").data)
        means = np.array(means)
        stds = np.array(stds)
        # print box results
        print("-----")
        print("budget results for second isotherm [kW/m]:")
        for name, data, sigma in zip(plot_terms, means, stds):
            print(name, f"{data/1e3:1.2f} +- {sigma/1e3:1.2f}")
        print("-----")
        print("terms for residuals:", sum_terms / 1e3)
        residual = np.sum(means[[1, 2, 3, 4, 6, 7]])
        print(f"residual: {residual/1e3:1.2f}")
        sigmas_squared = np.array(stds[[1, 2, 3, 4, 6, 7]]) ** 2
        combined_error = np.sqrt(np.sum(sigmas_squared))
        print(f"residual sigma: {combined_error/1e3:1.2f}")
        print("-----")
        tmp = ax.barh(
            x + addwidth * width,
            means / 1e3,
            width,
            color="#F06292",
            label="model 0.8$^{\circ}$C",
        )
        h.append(tmp[0])
        for xi, mi, si in zip(x + addwidth * width, means / 1e3, stds / 1e3):
            ax.hlines(y=xi, xmin=mi - si, xmax=mi + si, color="0.3", linewidth=2)
        addwidth += 1
    addwidth += 0.5

    # towyo energy budget results
    tyb = xr.open_dataset(cfg.obs.energy_budget)
    for year, ty in tyb.groupby("year"):
        tyterms = np.zeros_like(means)
        if refrho_sorted:
            tyterms[1] = ty.apesflux_div
            tyterms[2] = ty.keflux_div
            tyterms[4] = ty.vertical_pressure_work_sorted
            tyterms[3] = ty.horizontal_pressure_work_sorted
            tyterms[5] = ty.vertical_wave_flux
            tyterms[6] = ty.dissipation
            tyterms[7] = ty.bottom_dissipation
        else:
            tyterms[1] = ty.apeflux_div
            tyterms[2] = ty.keflux_div
            tyterms[4] = ty.vertical_pressure_work
            tyterms[3] = ty.horizontal_pressure_work
            tyterms[5] = ty.vertical_wave_flux
            tyterms[6] = ty.dissipation
            tyterms[7] = ty.bottom_dissipation
        tmp = ax.barh(x + addwidth * width, tyterms / 1e3, width, label=f"towyo {year}")
        h.append(tmp[0])
        addwidth += 1

    centerwidth = (addwidth - 1) * width / 2
    ax.spines["left"].set_visible(False)
    yticks = x[:-1] + width * (addwidth + 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set(ylim=[yticks[-1] + 1, yticks[0] - 1])
    ax.set(xlabel="Energy Flux [kW/m]")
    _add_bar_label_h(ax, "steadiness", 0, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "APE flux divergence", 1, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(ax, "KE flux divergence", 2, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(
        ax,
        "vertical pressure work\ndivergence $w'p'$",
        4,
        centerwidth=centerwidth,
        lr="l",
    )
    _add_bar_label_h(
        ax,
        "horizontal pressure work\ndivergence $v'p'$",
        3,
        centerwidth=centerwidth,
        offset=20,
        lr="r",
    )
    _add_bar_label_h(
        ax,
        "vertical internal wave\nflux divergence $w''p''$",
        5,
        centerwidth=centerwidth,
        lr="l",
    )
    _add_bar_label_h(ax, "turbulent dissipation", 6, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "bottom drag", 7, centerwidth=centerwidth, lr="l")
    ax.grid(True)
    ax.legend(loc=4)
    gv.plot.xsym(ax=ax)


def PlotEnergeticsLayerBudgetResultsHorizontalTwoIsothermsSourcePositive(
    cfg, refrho_sorted=True
):
    """Plot energy budget results with horizontal bars. This version includes
    pressure work terms v'p' and w'p'. Also, this version does not show the box
    results but shows results for two different isotherms instead. Energy
    sources are shown as positive and sinks as negative.

    Also, choose the way we calculate a reference density profile.

    Parameters
    ----------
    cfg : Box
        Config read from `config.yml` via :fun:`nslib.io.load_config`
    refrho_sorted : bool
        Calculate background density based on a sorted background state. Defaults to True.
    """
    # Read model energy budget results and concatenate them for plotting.
    B = xr.open_dataset(cfg.model.output.energy_budget_results_layer)
    B2 = xr.open_dataset(cfg.model.output.energy_budget_results_layer_08)
    if refrho_sorted:
        Epp = "EppSortedFluxDivergence"
        IWVert = "IWSortedFluxVertDivergence"
        IWHoriz = "IWSortedFluxHorizDivergence"
    else:
        Epp = "EppFluxDivergence"
        IWVert = "IWFluxVertDivergence"
        IWHoriz = "IWFluxHorizDivergence"
    plot_terms = [
        "Unsteadiness",
        Epp,
        "EkpFluxDivergence",
        IWHoriz,
        IWVert,
        "IWFluxVertDivergence_hp",
        "Dissipation",
        "BottomDrag",
    ]
    means = []
    stds = []
    for var in plot_terms:
        means.append(B[var].mean(dim="T").data)
        stds.append(B[var].std(dim="T").data)
    means = np.array(means)
    stds = np.array(stds)
    # print layer results
    print("-----")
    print("budget results for layer [kW/m]:")
    for name, data, sigma in zip(plot_terms, means, stds):
        print(name, f"{data/1e3:1.2f} +- {sigma/1e3:1.2f}")
    sum_terms = means[[1, 2, 3, 4, 6, 7]]
    print("terms for residuals:", sum_terms)
    residual = np.sum(means[[1, 2, 3, 4, 6, 7]])
    print(f"residual: {residual/1e3:1.2f}")
    sigmas_squared = np.array(stds[[1, 2, 3, 4, 6, 7]]) ** 2
    combined_error = np.sqrt(np.sum(sigmas_squared))
    print(f"residual sigma: {combined_error/1e3:1.2f}")
    print("-----")

    fig, ax = gv.plot.quickfig(fs=12, fgs=(5.5, 6))
    col = _linecolors(ax, option=0)
    x = np.arange(len(plot_terms))
    width = 0.15
    h = []
    tmp = ax.barh(x, -means / 1e3, width, color="C6", label="model 0.9$^{\circ}$C")
    h.append(tmp[0])
    for xi, mi, si in zip(x, means / 1e3, stds / 1e3):
        ax.hlines(y=xi, xmin=-mi - si, xmax=-mi + si, color="0.3", linewidth=2)
    addwidth = 1
    if B2 is not None:
        means = []
        stds = []
        for var in plot_terms:
            means.append(B2[var].mean(dim="T").data)
            stds.append(B2[var].std(dim="T").data)
        means = np.array(means)
        stds = np.array(stds)
        # print box results
        print("-----")
        print("budget results for second isotherm [kW/m]:")
        for name, data, sigma in zip(plot_terms, means, stds):
            print(name, f"{data/1e3:1.2f} +- {sigma/1e3:1.2f}")
        print("-----")
        print("terms for residuals:", sum_terms / 1e3)
        residual = np.sum(means[[1, 2, 3, 4, 6, 7]])
        print(f"residual: {residual/1e3:1.2f}")
        sigmas_squared = np.array(stds[[1, 2, 3, 4, 6, 7]]) ** 2
        combined_error = np.sqrt(np.sum(sigmas_squared))
        print(f"residual sigma: {combined_error/1e3:1.2f}")
        print("-----")
        tmp = ax.barh(
            x + addwidth * width,
            -means / 1e3,
            width,
            color="#F06292",
            label="model 0.8$^{\circ}$C",
        )
        h.append(tmp[0])
        for xi, mi, si in zip(x + addwidth * width, means / 1e3, stds / 1e3):
            ax.hlines(y=xi, xmin=-mi - si, xmax=-mi + si, color="0.3", linewidth=2)
        addwidth += 1
    addwidth += 0.5

    # towyo energy budget results
    tyb = xr.open_dataset(cfg.obs.output.energy_budget)
    for year, ty in tyb.groupby("year"):
        tyterms = np.zeros_like(means)
        if refrho_sorted:
            tyterms[1] = ty.apesflux_div
            tyterms[2] = ty.keflux_div
            tyterms[4] = ty.vertical_pressure_work_sorted
            tyterms[3] = ty.horizontal_pressure_work_sorted
            tyterms[5] = ty.vertical_wave_flux
            tyterms[6] = ty.dissipation
            tyterms[7] = ty.bottom_dissipation
        else:
            tyterms[1] = ty.apeflux_div
            tyterms[2] = ty.keflux_div
            tyterms[4] = ty.vertical_pressure_work
            tyterms[3] = ty.horizontal_pressure_work
            tyterms[5] = ty.vertical_wave_flux
            tyterms[6] = ty.dissipation
            tyterms[7] = ty.bottom_dissipation
        tmp = ax.barh(
            x + addwidth * width, -tyterms / 1e3, width, label=f"towyo {year}"
        )
        h.append(tmp[0])
        addwidth += 1

    centerwidth = (addwidth - 1) * width / 2
    ax.spines["left"].set_visible(False)
    yticks = x[:-1] + width * (addwidth + 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set(ylim=[yticks[-1] + 1, yticks[0] - 1])
    ax.set(xlabel="Energy Flux [kW/m]")
    _add_bar_label_h(ax, "steadiness", 0, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(ax, "APE flux divergence", 1, centerwidth=centerwidth, lr="l")
    _add_bar_label_h(ax, "KE flux divergence", 2, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(
        ax,
        "vertical pressure work\ndivergence $w'p'$",
        4,
        centerwidth=centerwidth,
        lr="r",
    )
    _add_bar_label_h(
        ax,
        "horizontal pressure work\ndivergence $v'p'$",
        3,
        centerwidth=centerwidth,
        offset=20,
        lr="l",
    )
    _add_bar_label_h(
        ax,
        "vertical internal wave\nflux divergence $w''p''$",
        5,
        centerwidth=centerwidth,
        lr="r",
    )
    _add_bar_label_h(ax, "turbulent dissipation", 6, centerwidth=centerwidth, lr="r")
    _add_bar_label_h(ax, "bottom drag", 7, centerwidth=centerwidth, lr="r")
    ax.grid(True)
    ax.legend(loc=3)
    gv.plot.xsym(ax=ax)


def PlotFormDrag():
    """Plot form drag."""
    a = nsl.io.load_towyos()
    ty = a["t12"]
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 5), constrained_layout=True, sharex=True
    )
    gv.plot.axstyle(ax1)
    gv.plot.axstyle(ax2)
    col = _linecolors(ax2, option=0)

    h = ax1.contourf(
        ty.dist,
        ty.z,
        ty.p_anom,
        levels=np.arange(-160, 170, 10),
        cmap="PuOr",
        antialiased=True,
    )
    for c in h.collections:
        c.set_rasterized(True)

    ax1.contour(
        ty.dist,
        ty.z,
        ty.gsw_sigma4,
        levels=ty.sg4bins,
        colors="0.5",
        linewidths=0.5,
    )
    ax1.contour(
        ty.dist,
        ty.z,
        ty.gsw_sigma4,
        levels=[_InterfaceSG4],
        colors="k",
        linewidths=0.75,
    )

    ax1.fill_between(
        ty.dist, ty.topo, ty.BottomLayerStartCTD + 1, color="k", alpha=0.25
    )
    ax1.fill_between(ty.dist, ty.topo, 10000, color="0.3")

    # ht = ax1.plot(ty.dist, ty.topo, "k")
    ax1.set_ylim([5200, 4050])
    ax1.set_yticks(np.arange(4200, 5200, 400))
    ax1.set_ylabel("Depth [m]")
    cb = plt.colorbar(h, ax=ax1, label=r"p$^\prime$ [N/m$^2$]", aspect=50)
    gv.plot.annotate_corner(
        "a) 2012 baroclinic pressure anomaly",
        ax=ax1,
        quadrant=1,
        addx=-0.02,
        addy=+0.03,
    )

    # bottom pressure was saved in model_form_drag.py
    mbp = xr.open_dataset("data/model_bottom_pressure.nc")
    mbp = mbp.swap_dims({"Y": "dist"})
    for g, p in mbp.bottom.groupby("T"):
        ax2.plot(p.dist, p + 60, color="k", alpha=0.01)
    (mbp.bottom.mean(dim="T") + 60).plot(ax=ax2, color="#F06292", label="model")
    # (mbp.full.mean(dim="T") + 60).plot(
    #     ax=ax2, color="#F06292", linestyle="--", label="model full"
    # )
    (a["t12"].swap_dims({"x": "dist"}).BottomPressure - 150).plot(
        ax=ax2, label="towyo 2012"
    )
    (a["t14"].swap_dims({"x": "dist"}).BottomPressure - 150).plot(
        ax=ax2, label="towyo 2014"
    )
    ax2.set(
        xlabel="Distance [km]",
        ylabel=r"$p_B$ [N/m$^2$]",
        xlim=(-2, 32),
        title="",
    )
    ax2.legend(loc=1)
    gv.plot.annotate_corner(
        "b) bottom pressure anomaly",
        ax=ax2,
        quadrant=1,
        addx=-0.02,
        addy=+0.03,
    )


def PlotFormDrag2(a):
    """Plot form drag. This second edition with an extra panel to show a
    comparison between full depth integral and integral below 4000m only. Also
    show pressure due to free ocean surface elevation and non-hydrostatic
    pressure. The top panel shows baroclinic pressure anomaly for one of the
    towyos. The center panel shows both towyo bottom pressure anomalies and
    time-mean model bottom pressur. The bottom model shows various model
    components for two time steps to illustrate a) that integrating density
    anomaly over only the lower part of the water column gives the right bottom
    pressure and b) that the model still baroclinically adjusts in the upper
    water column, thereby changing the surface pressure component. We thus do
    not further analyze the surface contribution.

    Parameters
    ----------
    a : dict
        Towyo data.
    fgs : tuple, optional
        Figure size (width, height). Defaults to (6, 4).
    """
    ty = a["t12"]
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, figsize=(8, 8), constrained_layout=True, sharex=True
    )
    gv.plot.axstyle(ax1)
    gv.plot.axstyle(ax2)
    gv.plot.axstyle(ax3)
    col = _linecolors(ax2, option=0)

    h = ax1.contourf(
        ty.dist,
        ty.z,
        ty.p_anom,
        levels=np.arange(-160, 170, 10),
        cmap="PuOr",
        antialiased=True,
    )
    for c in h.collections:
        c.set_rasterized(True)

    ax1.contour(
        ty.dist,
        ty.z,
        ty.gsw_sigma4,
        levels=ty.sg4bins,
        colors="0.5",
        linewidths=0.5,
    )
    ax1.contour(
        ty.dist,
        ty.z,
        ty.gsw_sigma4,
        levels=[_InterfaceSG4],
        colors="k",
        linewidths=0.75,
    )

    ax1.fill_between(
        ty.dist, ty.topo, ty.BottomLayerStartCTD + 1, color="k", alpha=0.25
    )
    ax1.fill_between(ty.dist, ty.topo, 10000, color="0.3")

    # ht = ax1.plot(ty.dist, ty.topo, "k")
    ax1.set_ylim([5200, 4050])
    ax1.set_yticks(np.arange(4200, 5200, 400))
    ax1.set_ylabel("Depth [m]")
    cb = plt.colorbar(h, ax=ax1, label=r"p$^\prime$ [N/m$^2$]", aspect=50)
    gv.plot.annotate_corner(
        "a) 2012 towyo baroclinic pressure anomaly",
        ax=ax1,
        quadrant=1,
        addx=-0.02,
        addy=+0.03,
    )

    # bottom pressure was saved in model_form_drag.py
    mbp = xr.open_dataset("data/model_bottom_pressure.nc")
    mbp = mbp.swap_dims({"Y": "dist"})
    # plot bottom pressure for each time step
    for g, p in mbp.full.groupby("T"):
        ax2.plot(p.dist, p + 60, color="k", alpha=0.01)
    # (mbp.bottom.mean(dim="T") + 60).plot(
    #     ax=ax2, color="#F06292", label="model bc"
    # )
    (mbp.full.mean(dim="T") + 60).plot(
        ax=ax2, color="#F06292", linestyle="-", label="model full"
    )
    (a["t12"].swap_dims({"x": "dist"}).BottomPressure - 150).plot(
        ax=ax2, label="towyo 2012"
    )
    (a["t14"].swap_dims({"x": "dist"}).BottomPressure - 150).plot(
        ax=ax2, label="towyo 2014"
    )
    ax2.set(
        xlabel="",
        ylabel=r"$p_B$ [N/m$^2$]",
        xlim=(-2, 32),
        title="",
    )
    ax2.legend(loc=1)
    gv.plot.annotate_corner(
        "b) bottom pressure anomaly",
        ax=ax2,
        quadrant=1,
        addx=-0.02,
        addy=+0.03,
    )

    for ti, alpha in zip([0, -1], [0.5, 1]):
        mbp.surface.isel(T=ti).plot(ax=ax3, alpha=alpha, color="r")
        mbp.bcfull.isel(T=ti).plot(ax=ax3, alpha=alpha, color="b")
    ax3.set(
        xlabel="Distance [km]",
        ylabel=r"$p_B$ [N/m$^2$]",
        xlim=(-2, 32),
        title="",
    )


def PlotMooredProfilerTimeSeries(mp):
    """Plot moored profiler time series.

    Parameters
    ----------
    mp : xr.Dataset
        MP data
    """

    def filter_mp(mp):
        v = mp.v.dropna(dim="depth", how="all").dropna(dim="time", how="all")
        v = v.sel(time=slice("2014-01-20 23:30", "2014-01-24 00:00"))
        v = v.dropna(dim="depth")

        tmin = np.datetime64(v.time[0].data, "h")
        tmax = np.datetime64(v.time[-1].data, "h")
        tnew = np.arange(tmin - np.timedelta64(1, "h"), tmax, dtype="datetime64[h]")

        V = v.interp(time=tnew)
        V = V.dropna(dim="time", how="all")

        N = 3  # Filter order
        Wn = (
            1 / 18.0
        )  # Cutoff frequency (1/50 would be 100 hours if sampling frequency is 1 hour)
        B, A = signal.butter(N, Wn, output="ba")
        # Second, apply the filter
        lpdata = signal.filtfilt(B, A, V, axis=0)
        # Bandpass as well
        B2, A2 = signal.butter(N, [1 / 18, 1 / 5], output="ba", btype="bandpass")
        bpdata = signal.filtfilt(B2, A2, V, axis=0)
        return V, lpdata, bpdata

    V, lpdata, bpdata = filter_mp(mp)
    pyt = V.time

    cmap = "RdBu_r"
    range = np.arange(-0.25, 0.27, 0.02)

    fig, ax = plt.subplots(nrows=3, figsize=(6, 8), sharey=False, sharex=True)
    for axi in ax:
        gv.plot.axstyle(axi)
    # pyt = gv.io.mtlb2datetime(tnew)
    h0 = ax[0].contourf(pyt, V["depth"], V.transpose(), range, cmap=cmap)
    h1 = ax[1].contourf(pyt, V["depth"], lpdata.transpose(), range, cmap=cmap)
    h2 = ax[2].contourf(pyt, V["depth"], bpdata.transpose(), range, cmap=cmap)
    for hi in [h0, h1, h2]:
        for c in hi.collections:
            # Fix for the white lines between contour levels
            c.set_edgecolor("face")
            # Rasterized contours in pdfs
            c.set_rasterized(True)
    # colorbar
    pad = 0.02
    width = 0.015
    extend = 0.15
    divider = make_axes_locatable(ax[1])
    dpos = divider.get_position()
    cax = fig.add_axes([0.15, 0.01, 0.7, 0.015])
    cb = plt.colorbar(
        h1,
        cax=cax,
        orientation="horizontal",
        label="v [m/s]",
        ticks=[-0.2, -0.1, 0, 0.1, 0.2],
    )

    lblx = ["a) full", "b) low-frequency [T>36 hrs]", "c) tidal [36>T>10 hrs]"]
    for ni, axi in enumerate(ax[0:3]):
        axi.set(ylim=(5100, 3900))
        axi.text(
            0.01,
            0.99,
            lblx[ni],
            transform=axi.transAxes,
            size=10,
            weight="bold",
        )
    ax[1].set(ylabel="Depth [m]")
    gv.plot.concise_date(ax[2])
    return V, lpdata, bpdata


def PlotTowyoSketch(integration_isotherm=1.02, box_linewidth=5, show_epsilon=True):
    # load 2012 towyo
    cfg = nsl.io.load_config()
    t = xr.open_dataset(cfg.obs.output.towyo_2012)
    t = t.swap_dims({"x": "dist"})

    # interpolate bathymetry and potential temperature to regular grid and
    # smooth for cartoon style plot
    smooth_over_dist_pts = 10
    smooth_over_z_m = 50
    distnew = np.arange(0, 30.1, 0.1)
    topo = t.topo.interp(dist=distnew)
    th = t.th.interp(dist=distnew)
    clevels_th = np.arange(0.72, 1.08, 0.04)
    ths = th.rolling(
        dict(dist=smooth_over_dist_pts, z=smooth_over_z_m), center=True
    ).mean()
    topos = topo.rolling(dist=smooth_over_dist_pts, center=True).mean()

    # select region to plot
    distmax = 18
    t0 = ths.sel(dist=1, method="nearest")
    t1 = ths.sel(dist=distmax, method="nearest")
    thsa = ths.where((ths.dist > 1) & (ths.dist <= distmax), drop=True)
    toposa = topos.where((ths.dist > 1) & (ths.dist <= distmax), drop=True)

    ths_mid = ths.sel(
        dist=distmax / 2,
        method="nearest",
    )
    ths_mid_z = ths_mid.where(ths_mid < integration_isotherm, drop=True).z.min().data

    fig, ax = gv.plot.quickfig(w=7, h=4)

    # show turbulent dissipation
    if show_epsilon:
        ax.pcolormesh(
            t.dist,
            t.z,
            np.log10(t.eps),
            cmap="magma_r",
            vmin=-10,
            vmax=-4,
            alpha=0.1,
        )
    # show bathymetry
    ax.fill_between(topo.dist, topos, 6000, color="0.7")
    # show isotherms
    ths.plot.contour(
        levels=clevels_th,
        colors="0.5",
        linewidths=1,
    )

    # plot integration volume
    for ti in [t0, t1]:
        ax.vlines(
            ti.dist,
            topos.sel(dist=ti.dist, method="nearest"),
            ti.where(ti <= integration_isotherm, drop=True).z.min(),
            "k",
            linewidths=box_linewidth,
        )
    thsa.plot.contour(
        levels=[integration_isotherm], colors="k", linewidths=box_linewidth
    )
    toposa.plot(linewidth=box_linewidth, linestyle="--", color="k")

    # horizontal flux of kinetic and potential energy
    for ti in [t0, t1]:
        ax.text(
            ti.dist,
            4500,
            r"$v^\prime E_k^\prime$, $v^\prime E_p^\prime$",
            ha="center",
            va="baseline",
            color="w",
            fontsize=14,
            fontweight="normal",
            bbox=dict(boxstyle="rarrow, pad=0.4", fc="k", ec="k"),
        )
    # horizontal wave energy flux
    for ti in [t0, t1]:
        ax.text(
            ti.dist,
            4700,
            r"$v^\prime p^\prime$",
            ha="center",
            va="baseline",
            color="w",
            fontsize=14,
            fontweight="normal",
            bbox=dict(boxstyle="rarrow, pad=0.4", fc="k", ec="k"),
        )
    # vertical wave energy flux pressure work
    ax.text(
        distmax / 2 - distmax / 10,
        ths_mid_z,
        r"$w' p'$",
        ha="center",
        va="center",
        color="w",
        fontsize=14,
        fontweight="normal",
        rotation=90,
        bbox=dict(boxstyle="rarrow, pad=0.4", fc="k", ec="k"),
    )
    ax.text(
        distmax / 2 + distmax / 10,
        ths_mid_z,
        r"$w'' p''$",
        ha="center",
        va="center",
        color="w",
        fontsize=14,
        fontweight="normal",
        rotation=90,
        bbox=dict(boxstyle="rarrow, pad=0.4", fc="0.5", ec="0.5", alpha=0.9),
    )
    # bottom friction
    ax.text(
        distmax / 2,
        topo.interp(dist=10),
        r"$D^\prime$",
        ha="center",
        va="center",
        color="w",
        fontsize=14,
        fontweight="normal",
        bbox=dict(boxstyle="Round4, pad=0.4", fc="k", ec="k"),
    )
    # interior turbulent dissipation
    ax.text(
        distmax / 2 - 2,
        4600,
        r"$\rho \, \varepsilon$",
        ha="center",
        va="center",
        color="w",
        fontsize=14,
        fontweight="normal",
        bbox=dict(boxstyle="Round4, pad=0.4", fc="k", ec="k"),
    )

    ax.set(ylim=(5100, 4000), xlim=(0, 22), ylabel="", xlabel="")
    for spi in ax.spines:
        ax.spines[spi].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def PlotMomentumFluxes(a, momfi):
    fontsize = 12
    fig, ax2, ax = gv.plot.newfigyy(width=5, height=6)
    for axi in [ax, ax2]:
        ax = gv.plot.axstyle(ax, fontsize=fontsize)
    col = _linecolors(ax, option=0)
    h1 = []
    # towyo integrated fluxes:
    for k, t in a.items():
        htmp = gv.plot.vstep(t.mf2_along_isopycnal, t.sg4bins, label="20" + k[1:])
        h1.append(htmp[0])
    h = gv.plot.vstep(
        momfi.mean(dim="time").data, momfi.isot, ax=ax2, color="C6", label="model"
    )
    h1.append(
        h[0],
    )
    ax.legend(h1, ["towyo 2012", "towyo 2014", "model"], fontsize=11)
    ax2.hlines(0.9, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
    ax2.hlines(0.8, -1e5, 1e5, color="0.2", linestyle="--", zorder=1, alpha=0.5)
    ax2.annotate(
        "interface",
        xy=(-3.8e4, 0.9),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )
    ax2.annotate(
        "model 0.8$^{\circ}$C interface",
        xy=(-3.8e4, 0.8),
        xytext=(0, 5),
        textcoords="offset points",
        zorder=10,
        fontsize=fontsize,
    )
    # ax.spines["left"].set_position(("data", 0))
    ax.set(
        ylim=(45.99, 45.92),
        xlim=(-4e4, 0.3e4),
        xlabel=r"$\int w^\prime p^\prime\ dy$ [W/m]",
        ylabel=r"$\sigma_4$ [kg/m$^3$ - 1000]",
    )
    ax2.set(
        ylim=(0.65, 1.0),
        ylabel=r"$\theta$ [$^{\circ}$C]",
        xlabel=r"$\int \rho v^\prime w^\prime\ dy$ [N/m]",
    )
    ax2.spines["right"].set_visible(True)
    almost_black = "#262626"
    spine = "right"
    ax2.spines[spine].set_linewidth(0.5)
    ax2.spines[spine].set_color(almost_black)
    ax2.spines[spine].set_position(("outward", 5))
    ax2.yaxis.label.set_color(almost_black)
    ax2.yaxis.label.set_size(fontsize)
    ax2.yaxis.offsetText.set_fontsize(fontsize)
    # ax2.grid(False)
    ax2.vlines(0, -10, 10, color=almost_black, linewidth=1)


# helper functions
_InterfaceSG4 = 45.94


def _plot_interface(ty, ax):
    r"""Plot upper interface ($\sigma_4$) of dense plume in towyo  data.

    Parameters
    ----------
    ty :
        Towyo dataset
    ax :
        Axis
    """
    ty.gsw_sigma4.plot.contour(
        ax=ax, levels=[_InterfaceSG4], colors="k", linewidths=0.5
    )


def _linecolors(ax, option=0):
    """

    Parameters
    ----------
    ax : axis

    option : {0}
        Color option. Defaults to 0.

    Returns
    -------
    col : list
        List with colors

    Notes
    -----
    option=0: blue and purple
    """

    if option == 0:
        col = ["#0277BD", "#6A1B9A"]
        c = mpl.cycler(color=col)
        ax.set_prop_cycle(c)
    return col


def _add_bar_label_h(ax, label, y, offset=10, centerwidth=0, lr="l"):
    if lr == "l":
        offset *= -1
        ha = "right"
    else:
        ha = "left"
    ax.annotate(
        label,
        (0, y + centerwidth),
        (offset, 0),
        textcoords="offset points",
        ha=ha,
        va="center",
        fontweight="normal",
        backgroundcolor="w",
    )


def load_colors():
    col1 = [
        (0.0, 0.6, 0.9019607843137255),
        (0.07058823529411765, 0.1450980392156863, 0.35294117647058826),
        (0.9490196078431372, 0.2196078431372549, 0.0784313725490196),
        (0.8745098039215686, 0.7176470588235294, 0.5450980392156862),
        (0.7137254901960784, 0.7647058823529411, 0.7725490196078432),
    ]
    col2 = [
        "#332288",
        "#88CCEE",
        "#44AA99",
        "#117733",
        "#999933",
        "#DDCC77",
        "#CC6677",
        "#882255",
        "#AA4499",
    ]
    return col1, col2
