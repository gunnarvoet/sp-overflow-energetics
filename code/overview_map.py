#!/usr/bin/env python
# coding: utf-8

"""
Plot overview map for Samoan Passage Northern Sill paper.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gvpy as gv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Polygon

import nslib as nsl


def make_colormap(seq):
    """
    Converts a sequence of RGB tuples containing floats in the interval (0,1).
    For some reason LinearSegmentedColormap cannot take an alpha channel,
    even though matplotlib colourmaps have one.
    """
    from matplotlib.colors import LinearSegmentedColormap

    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {"red": [], "green": [], "blue": []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict["red"].append([item, r1, r2])
            cdict["green"].append([item, g1, g2])
            cdict["blue"].append([item, b1, b2])
    return LinearSegmentedColormap("CustomMap", cdict)


def add_alpha(cmap, alpha=None):
    """
    Add an alpha channel (opacity) to a colourmap. Uses a ramp by default.
    Pass an array of 256 values to use that. 0 = transparent; 1 = opaque.
    """
    from matplotlib.colors import ListedColormap

    cmap4 = cmap(np.arange(cmap.N))
    if alpha is None:
        alpha = np.linspace(1, 0, cmap.N)
    cmap4[:, -1] = alpha
    return ListedColormap(cmap4)


def smooth_topo(topo, sigma=2):
    """
    Smooth topography using a gaussian filter in 2D.
    """
    import scipy.ndimage as ndimage

    stopo = ndimage.gaussian_filter(topo, sigma=(sigma, sigma), order=0)
    return stopo


def generate_hill_shade(topo, root, azdeg=275, altdeg=145):
    """
    Generate image with hill shading.
    """
    from matplotlib.colors import LightSource

    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    bumps = ls.hillshade(topo) ** root  # Taking a root backs it off a bit.
    return bumps


def scaleyticklabel(x, pos):
    """scale the ytick label by 1e-3"""
    return "{0:g}".format(x / 1000)


def add_colorbar(h, ax, ticks):
    """Add colorbar to map and make a few modifications"""
    cb = plt.colorbar(
        h,
        pad=0.04,
        fraction=0.04,
        shrink=0.7,
        label="Depth [km]",
        ticks=ticks,
        ax=ax,
        format=mpl.ticker.FuncFormatter(scaleyticklabel),
        orientation="horizontal",
    )
    cb.ax.invert_yaxis()
    cb.solids.set_rasterized(True)
    cb.solids.set_antialiased(True)
    cb.outline.set_visible(False)
    return cb


def add_subplot_letter(ax, letter, yloc, col):
    ax.text(
        0.03,
        yloc,
        letter,
        transform=ax.transAxes,
        horizontalalignment="left",
        fontweight="bold",
        fontsize=12,
        color=col,
        zorder=20,
    )


def show_equator(ax, color="0.8", lw=1):
    import matplotlib.ticker as mticker

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=lw,
        color=color,
        linestyle="-",
        zorder=30,
    )
    gl.ylocator = mticker.FixedLocator([0])


def read_data():
    print("loading towyo data...")
    a = nsl.io.load_towyos()
    print("loading bathymetry...")
    ss = gv.ocean.smith_sandwell(subsample=40)
    ss = ss.load()
    elon = ss.lon
    elat = ss.lat
    etopo = ss.data
    etopo = etopo
    bathy = nsl.io.load_sp_bathy()
    blon = bathy.lon
    blat = bathy.lat
    btopo = bathy.z
    return a, elon, elat, etopo, blon, blat, btopo


class OverviewMap:
    def __init__(self):
        """OverviewMap class for plotting various versions of an overview map.

        Reads towyo and bathymetry data when initialized.
        """
        (
            self.a,
            self.elon,
            self.elat,
            self.etopo,
            self.blon,
            self.blat,
            self.btopo,
        ) = read_data()
        self.etopo = -1 * self.etopo

    def shading_calcs(self):
        self.topo_extent = (
            self.blon.min(),
            self.blon.max(),
            self.blat.max(),
            self.blat.min(),
        )
        self.etopo_extent = (
            self.elon.min(),
            self.elon.max(),
            self.elat.max(),
            self.elat.min(),
        )

        self.kmap = make_colormap([(0, 0, 0)])
        self.kmap4 = add_alpha(self.kmap)

        self.smoothbtopo = smooth_topo(self.btopo, sigma=10)
        self.smoothbtopo2 = smooth_topo(self.btopo, sigma=4)

        self.smoothetopo = smooth_topo(self.etopo, sigma=2)

        self.smoothbumps = generate_hill_shade(self.smoothbtopo, 0.2)
        self.smoothbumps2 = generate_hill_shade(self.smoothbtopo2, 0.4)

        tmp = self.smoothetopo
        tmp[tmp < 0] = 0
        self.smoothetoposhading = generate_hill_shade(
            self.smoothetopo, root=0.15, azdeg=275, altdeg=135
        )

    def globe(self, ax, cmap="ocean4jbm_r"):
        self.shading_calcs()
        print("plotting globe...")
        h = ax.pcolormesh(
            self.elon,
            self.elat,
            self.etopo,
            vmin=0,
            vmax=8000,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            zorder=9,
            rasterized=True,
        )
        ax.imshow(
            self.smoothetoposhading,
            extent=self.etopo_extent,
            cmap=self.kmap4,
            alpha=0.3,
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND, edgecolor=None, facecolor="0.1", zorder=12)

        show_equator(ax, lw=0.75)

        tlon, tlat = -165, -9
        splabel = ax.text(
            tlon,
            tlat,
            "Samoan\nPassage",
            color="w",
            ha="left",
            va="center",
            size=10,
            weight="bold",
            zorder=15,
            transform=ccrs.PlateCarree(),
        )
        # plot zoom area
        lats = np.array([-10.5 - 0.5, -10.5 - 0.5, -7.2 + 0.5, -7.2 + 0.5])
        lons = np.array([-171 - 0.5, -168 + 0.5, -168 + 0.5, -171 - 0.5])
        xy = np.column_stack([lons, lats])
        poly = Polygon(
            xy,
            facecolor=None,
            edgecolor="w",
            zorder=13,
            linewidth=1,
            fill=False,
            transform=ccrs.PlateCarree(),
        )
        hp = ax.add_patch(poly)
        # colorbar
        cbyticks = [2500, 5000, 7500]
        cb = add_colorbar(h, ax, cbyticks)

    def samoan_passage(self, ax, cmap="ocean4jbm_r"):
        print("plotting SP...")

        MinDepth = 3000
        MaxDepth = 6000

        h = ax.contourf(
            self.blon,
            self.blat,
            self.smoothbtopo2,
            np.arange(MinDepth, MaxDepth, 100),
            cmap=cmap,
            vmin=MinDepth,
            vmax=MaxDepth + 500,
            extend="both",
            zorder=9,
        )
        for c in h.collections:
            c.set_rasterized(True)
            c.set_edgecolor("face")

        ax.imshow(
            self.smoothbumps,
            extent=self.topo_extent,
            cmap=self.kmap4,
            alpha=0.5,
            zorder=10,
        )

        # contour depth
        h2 = ax.contour(
            self.blon,
            self.blat,
            self.smoothbtopo2,
            np.arange(MinDepth, MaxDepth, 500),
            colors="0.1",
            linewidths=0.25,
            zorder=11,
        )
        for c in h2.collections:
            c.set_rasterized(True)

        ax.set(
            xlim=(-171, -168),
            xticks=np.arange(-171, -167, 2),
            ylim=(-10.5, -7.2),
            yticks=np.arange(-10, -7, 1),
        )
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.tick_top()

        cbyticks = np.arange(3, 7) * 1e3
        add_colorbar(h, ax, cbyticks)

        # show zoom area
        lats = np.array([-8.35, -8.35, -7.9, -7.9])
        lons = np.array([-168.9, -168.4, -168.4, -168.9])
        xy = np.column_stack([lons, lats])
        poly = Polygon(
            xy,
            facecolor="none",
            edgecolor="w",
            alpha=0.9,
            zorder=12,
            linewidth=2,
        )
        hp = ax.add_patch(poly)
        h, ht = gv.maps.cartopy_scale_bar(
            ax,
            (0.1, 0.06),
            50,
            color="w",
            plot_kwargs=dict(zorder=15),
            text_kwargs=dict(zorder=16),
        )

    def northern_sill(self, ax, cmap="ocean4jbm_r"):
        print("plotting Northern Sill region...")
        MaxDepth = 5500
        MinDepth = 4000

        # Extract towyo positions
        lon = []
        lat = []
        for k, ty in self.a.items():
            lon.append(ty.lon.values)
            lat.append(ty.lat.values)

        h = ax.contourf(
            self.blon,
            self.blat,
            self.btopo,
            np.arange(MinDepth, MaxDepth, 50),
            cmap=cmap,
            vmin=MinDepth,
            vmax=MaxDepth + 200,
            extend="both",
            zorder=9,
        )
        for c in h.collections:
            c.set_rasterized(True)
            c.set_edgecolor("face")

        ax.imshow(
            self.smoothbumps2,
            extent=self.topo_extent,
            cmap=self.kmap4,
            alpha=0.5,
            zorder=10,
        )

        h2 = ax.contour(
            self.blon,
            self.blat,
            self.btopo,
            np.arange(MinDepth, MaxDepth, 100),
            colors="0.1",
            linewidths=0.25,
            zorder=11,
        )
        for c in h2.collections:
            c.set_rasterized(True)

        ax.set(
            xlim=(-168.9, -168.4),
            xticks=np.array([-168.75, -168.5]),
            ylim=(-8.35 - 0.024, -7.9 + 0.024),
            yticks=np.array([-8.25, -8]),
        )
        from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.xaxis.tick_top()

        # Plot towyo lines
        labels = ["2012", "2014"]
        htowyo = []
        for i, loni, lati, label in zip(["#FF6D00", "#FFCA28"], lon, lat, labels):
            htmp = ax.plot(
                loni,
                lati,
                color=i,
                linewidth=2,
                alpha=0.7,
                zorder=15,
                label=label,
            )
            htowyo.append(htmp[0])
        leg = ax.legend(htowyo, labels, loc=(0.75, 0.6))
        leg.set_zorder(17)

        # Plot T1 mooring location
        sp14moorings = xr.open_dataset(
            "/Users/gunnar/Projects/hyprsp/py/sp-mooring-locations/sp14_mooring_locations.nc"
        )
        T1 = sp14moorings.sel(mooring="T1")
        ax.plot(
            T1.lon,
            T1.lat,
            marker="o",
            markersize=3,
            color="w",
            zorder=15,
            label="T1",
        )
        ax.annotate(
            "T1",
            xy=(T1.lon, T1.lat),
            xytext=(7, -7),
            textcoords="offset points",
            color="w",
            zorder=15,
        )

        cbyticks = [5500, 5000, 4500, 4000]
        add_colorbar(h, ax, cbyticks)

        h, ht = gv.maps.cartopy_scale_bar(
            ax,
            (0.87, 0.92),
            5,
            color="w",
            plot_kwargs=dict(zorder=15),
            text_kwargs=dict(zorder=16),
        )

    def plot_globe(self):
        """Plot globe by itself."""
        projection = ccrs.Orthographic(central_longitude=-170, central_latitude=-10)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10, 10),
            subplot_kw={"projection": projection},
        )
        self.globe(ax)

    def plot_samoan_passage(self):
        """Plot Samoan Passage region by itself."""
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10, 10),
            subplot_kw={"projection": projection},
        )
        self.samoan_passage(ax)

    def plot_northern_sill(self):
        """Plot Northern Sill region by itself."""
        projection = ccrs.PlateCarree()
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(10, 10),
            subplot_kw={"projection": projection},
        )
        self.northern_sill(ax)

    def plot_overview_map(self):
        """Plot overview map for Northern Sill paper."""
        nr = 1
        nc = 12
        cmap = "ocean4jbm_r"
        fig = plt.figure(figsize=(13.5, 5))

        mapax1 = plt.subplot2grid(
            (nr, nc),
            (0, 0),
            rowspan=1,
            colspan=3,
            projection=ccrs.Orthographic(central_longitude=-170, central_latitude=-10),
        )
        mapax2 = plt.subplot2grid(
            (nr, nc),
            (0, 4),
            rowspan=1,
            colspan=3,
            projection=ccrs.PlateCarree(),
        )
        mapax3 = plt.subplot2grid(
            (nr, nc),
            (0, 8),
            rowspan=1,
            colspan=3,
            projection=ccrs.PlateCarree(),
        )
        self.globe(mapax1, cmap=cmap)
        self.samoan_passage(mapax2, cmap=cmap)
        self.northern_sill(mapax3, cmap=cmap)

        # Subplot Letters
        ax = mapax1, mapax2, mapax3
        letters = "a", "b", "c"
        ylocs = 0.96, 0.92, 0.92
        colors = "k", "w", "w"
        for axi, letter, yloc, col in zip(ax, letters, ylocs, colors):
            add_subplot_letter(axi, letter, yloc, col)

        # plt.savefig(
        #     "fig/map_overview.png", dpi=300, bbox_inches="tight", pad_inches=0.1
        # )


if __name__ == "__main__":
    om = OverviewMap()
    om.shading_calcs()
    om.plot_overview_map()
    nsl.io.save_png("map_overview")
