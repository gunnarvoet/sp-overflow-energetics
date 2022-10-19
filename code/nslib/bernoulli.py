#!/usr/bin/env python
# coding: utf-8

"""
Calculate Bernoulli flux for model and observations.
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
from abc import ABC, abstractmethod


class Bernoulli(ABC):

    upper_layer_depth_limit = 4167
    rho0 = 1000
    _volume_transport = None
    _form_drag_dissipation = None

    @property
    def volume_transport(self):
        if self._volume_transport is None:
            self._volume_transport = self.d * self.v
        return self._volume_transport

    @abstractmethod
    def calculate_drho(self):
        pass

    @abstractmethod
    def calculate_d(self):
        pass

    @abstractmethod
    def calculate_h(self):
        pass

    @abstractmethod
    def calculate_v(self):
        pass

    def calculate_bernoulli_function(self):
        # Bernoulli function along the flow
        self.gprime = self.drho / self.rho0 * 9.81
        self.gprime.name = "gprime"
        self.gprime.attrs["long_name"] = r"g$^\prime$"
        self.gprime.attrs["units"] = r"m/s$^2$"
        # Convert to units of kW/m
        conversion = self.rho0 / 1000

        self.B_ke = self.v**3 * self.d / 2 * conversion
        self.B_ke.name = "KE"
        self.B_ke.attrs["long_name"] = "KE"
        self.B_ke.attrs["units"] = "kW/m"
        self.B_ke.attrs["color"] = "C0"

        self.B_pe_thickness = self.v * self.gprime * self.d**2 * conversion
        self.B_pe_thickness.name = "PE thickness"
        self.B_pe_thickness.attrs["long_name"] = "PE thickness"
        self.B_pe_thickness.attrs["units"] = "kW/m"
        self.B_pe_thickness.attrs["color"] = "C4"

        self.B_pe_elevation = (
            self.v * self.gprime * self.h * self.d * conversion
        )
        self.B_pe_elevation.name = "PE elevation"
        self.B_pe_elevation.attrs["long_name"] = "PE elevation"
        self.B_pe_elevation.attrs["units"] = "kW/m"
        self.B_pe_elevation.attrs["color"] = "C6"

        self.B = self.B_ke + self.B_pe_thickness + self.B_pe_elevation
        self.B.name = "Bernoulli flux"
        self.B.attrs["long_name"] = "Bernoulli flux"
        self.B.attrs["units"] = "kW/m"
        self.B.attrs["color"] = "k"

    def plot_volume_transport(self, ax=None):
        if ax is None:
            fig, ax = gv.plot.quickfig()
        self.volume_transport.plot(ax=ax)
        ax.set(xlim=(-10, 50), ylim=(0, 200), title="")
        return ax

    def plot_bernoulli_components(self, ax=None, smooth=5):
        if ax is None:
            fig, ax = gv.plot.quickfig()

        def plot_component(component, ax):
            component.plot(
                ax=ax,
                alpha=0.4,
                color=component.attrs["color"],
            )
            component.rolling(dist=smooth, center=True).mean().plot(
                ax=ax, label=component.name, color=component.attrs["color"]
            )

        components = [
            self.B,
            self.B_ke,
            self.B_pe_thickness,
            self.B_pe_elevation,
        ]
        # Average over model time if present
        if 'T' in self.B.dims:
            components = [ci.mean(dim='T', keep_attrs=True) for ci in components]

        [plot_component(component, ax) for component in components]
        ax.legend()
        ax.set(ylabel="energy flux [kW/m]")
        return ax

    def calculate_single_layer_form_drag_dissipation(self, Df, max_dist=2):
        """Calculate Larry's parameterization for form drag dissipation.

        Parameters
        ----------
        Df : float
            Form drag [N/m]
        max_dist : float
            Average 1.5 layer parameters from km 0 to this distance [km].

        Returns
        -------
        dissipation_estimate : float
            Estimate of integrated epsilon times rho [W/m].
        """
        print('form drag Df [N/m]:', Df)
        # Calculate 1.5 layer parameters from km 0 to max_dist
        da = self.d.where(self.data.dist<2).mean(dim='dist').data
        va = self.v.where(self.data.dist<2).mean(dim='dist').data
        gpr = self.gprime.where(self.data.dist<2).mean(dim='dist').data
        rho = self.rho0

        dissipation_estimate = 0.9744 * va * Df + 0.8608 * va * Df**2 / (rho * gpr * da**2)

        if 'T' not in self.data.dims:
            print(f'upstream velocity [m/s]: {va:1.2f}')
            print(f'layer thickness da [m]: {da:1.0f}')
            print(f'g\' [kg/m^3]: {gpr:1.1e}')
            print(f'dissipation [W/m] {dissipation_estimate:1.0f}')

        return dissipation_estimate

    def calculate_entrainment_velocity(self, smooth=40):
        """Calculate entrainment velocity for the layer.

        Parameters
        ----------
        smooth : int
            Smooth by this much times 200m before differentiating.

        Returns
        -------
        we : xr.DataArray
            Entrainment velocity along the flow.
        """
        q = self.volume_transport
        # Interpolate to evenly spaced distance vector so we can do some proper smoothing.
        dist_even = np.arange(-10, 40, 0.2)
        qi = q.interp(dist=dist_even)
        # smooth over 8 km (40*200m)
        qis = qi.rolling(dist=smooth, center=True).mean()
        we = qis.differentiate(coord='dist', edge_order=1)
        # account for distance in kilometers
        we = we / 1e3
        return we

class ModelBernoulli(Bernoulli):

    _mask_lower = None
    _mask_upper = None

    def __init__(self, data, interface=0.8):
        self.data = data
        self.interface = interface
        self.calculate_d()
        self.calculate_h()
        self.calculate_v()
        self.calculate_drho()
        self.calculate_bernoulli_function()

    @property
    def mask_lower(self):
        if self._mask_lower is None:
            self._mask_lower = self.data.th < self.interface
        return self._mask_lower

    @property
    def mask_upper(self):
        if self._mask_upper is None:
            self._mask_upper = (self.data.th > self.interface) & (
                self.data.Z > -1 * self.upper_layer_depth_limit
            )
        return self._mask_upper

    def calculate_d(self):
        self.d = (
            (self.data.HFacC * self.data.drF)
            .where(self.mask_lower)
            .sum(dim="Z")
        )

    def calculate_h(self, reference_depth=5282):
        # Bottom elevation h.
        depth = self.data.Depth.isel(Z=0)
        self.h = np.abs(depth - reference_depth)

    def calculate_v(self):
        # Layer mean velocity.
        self.v = (
            (self.data.HFacC * self.data.VC) * (self.data.HFacC * self.data.drF)
        ).where(self.mask_lower).sum(dim="Z") / self.d

    def calculate_drho(self):
        rho = self.data.rho * self.data.BathyMask
        rho_l = rho.where(self.mask_lower).mean(dim="Z")
        rho_u = rho.where(self.mask_upper).mean(dim="Z")
        self.drho = rho_l - rho_u


class TowyoBernoulli(Bernoulli):

    _mask_lower = None
    _mask_upper = None

    def __init__(self, data, interface=45.94):
        self.data = data.swap_dims(x="dist")
        self.interface = interface
        self.calculate_d()
        self.calculate_h()
        self.calculate_v()
        self.calculate_drho()
        self.calculate_bernoulli_function()

    @property
    def mask_lower(self):
        if self._mask_lower is None:
            self._mask_lower = self.data.gsw_sigma4 > self.interface
        return self._mask_lower

    @property
    def mask_upper(self):
        if self._mask_upper is None:
            self._mask_upper = (self.data.gsw_sigma4 < self.interface) & (
                self.data.z > self.upper_layer_depth_limit
            )
        return self._mask_upper

    def calculate_d(self):
        # Calculate layer thickness
        def sg4_depth(s, sg4=self.interface):
            tmp = s.dropna(dim="z")
            return tmp.z.where(tmp < self.interface, drop=True).max()

        sg4z = self.data.gsw_sigma4.groupby("dist").map(sg4_depth)
        self.d = self.data.topo - sg4z

    def calculate_h(self, reference_depth=5282):
        # Bottom elevation h.
        depth = self.data.topo
        self.h = np.abs(depth - reference_depth)

    def calculate_v(self):
        # Layer mean velocity.
        self.v = self.data.v.where(self.mask_lower).mean(dim="z")

    def calculate_drho(self):
        rho_u = self.data.gsw_sigma4.where(self.mask_upper).mean(dim="z")
        rho_l = self.data.gsw_sigma4.where(self.mask_lower).mean(dim="z")
        self.drho = rho_l - rho_u
        self.drho.name = "drho"
        self.drho.attrs["long_name"] = r"$\Delta \rho$"
        self.drho.attrs["units"] = r"kg/m$^3$"


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
        thickness = (c.HFacC * c.drF).where(mask).sum(dim="Z")
    else:
        # find depth of interface and subtract from bottom depth for layer thickness
        z = c.Z
        interface_z = []
        for dd, x in c.th.groupby("dist"):
            #     xi = ~np.isnan(x)
            interface_z.append(
                sp.interpolate.interp1d(x, z, axis=0, bounds_error=False)(
                    model_interface
                )
            )

        thickness = np.array(interface_z) + depth
    d = thickness
    # Layer mean velocity.
    v = ((c.HFacC * c.VC) * (c.HFacC * c.drF)).where(mask).sum(dim="Z") / d
    # Bernoulli function along the flow
    gprime = drho / 999 * 9.81
    B = v**3 * d / 2 + v * gprime * (d**2 + h * d)
    # Convert to units of kW/m
    B = rho0 * B / 1000

    return B
