"""
Read and write data and figures.
"""

import collections.abc
import warnings
from collections import OrderedDict
from pathlib import Path

import gsw
import gvpy as gv
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
import xarray as xr
import yaml
from box import Box
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore", message="Mean of empty slice")


def load_config() -> Box:
    """Load the yaml config file.

    Returns
    -------
    cfg : Box
        Config parameters dictionary with dot access.
    """

    def find_config_file():
        # first look in current directory
        cwd = Path(".").absolute()
        files = list(cwd.glob("config.yml"))
        if len(files) == 1:
            cfile = files[0]
        else:
            # otherwise go through parent directories
            parents = list(Path.cwd().parents)
            for pi in parents:
                files = list(pi.glob("config.yml"))
                if len(files) == 1:
                    cfile = files[0]
        return cfile

    configfile = find_config_file()
    with open(configfile, "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
    cfg.path.root = configfile.parent

    # Convert paths to Path objects
    # cfg.path.root = Path(cfg.path.root)
    cfg.path.data = cfg.path.root.joinpath(cfg.path.data)
    cfg.path.fig = cfg.path.root.joinpath(cfg.path.fig)

    def replace_variables(dict_in, var, rootpath):
        d = dict_in.copy()
        n = len(var)
        for k, v in d.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = replace_variables(d.get(k, {}), var, rootpath)
            elif isinstance(v, str):
                if v.startswith(var + "/"):
                    d[k] = rootpath.joinpath(v[n + 1 :])
            else:
                d[k] = v
        return d

    # Replace variables from the yaml file.
    cfg = replace_variables(cfg, "$data", cfg.path.data)
    cfg = replace_variables(cfg, "$input", cfg.path.input)
    cfg = replace_variables(cfg, "$output", cfg.path.output)

    return cfg


def print_config(print_values=False):
    config = load_config()
    gv.misc.pretty_print(config, print_values=print_values)


def load_sp_bathy() -> xr.Dataset:
    """Load Samoan Passage multibeam bathymetry.

    Returns
    -------
    bathy : xr.Dataset
        Bathymetry

    Notes
    -----
    Path to bathy netcdf file is set in config.yml.
    """
    cfg = load_config()
    bathy = xr.open_dataset(cfg.obs.input.bathy)
    # rename depth to z as that's the variable name that was used throughout
    # the analysis
    bathy = bathy.rename(dict(depth="z"))
    return bathy


def load_towyo(year: int, cast: int) -> xr.Dataset:
    """Read processed towyo data.

    Path to data is defined in config file.

    Parameters
    ----------
    year : {2012, 2014}
        Towyo year
    cast : int
        castnumber as assigned during the cruise

    Returns
    -------
    t : xr.Dataset
        Towyo data
    """
    cfg = load_config()
    path = cfg.obs.input.towyo.joinpath(f"sp_{year}_towyo_{cast:03}.nc")
    t = xr.open_dataset(path)
    # Rename variable depth to z (as all the older code of this project still
    # works this way...).
    t = t.rename(depth="z")
    return t


def load_mp_t1() -> xr.Dataset:
    """Load MP T01.

    Returns
    -------
    mp : xr.Dataset
        MP data
    """
    cfg = load_config()
    mp = xr.open_dataset(cfg.obs.input.mp_t1)
    return mp


def write_towyo_results(a, file):
    parameters = dict(
        FormDrag=("{v:1.1e}", "FormDrag"),
        FormDragStress=("{v:1.1f}", "FormDragStress"),
    )

    out = dict()
    for (k, ty), ab in zip(a.items(), ["A", "B"]):
        for key, (fmt, value) in parameters.items():
            # extract value from towyo dataset
            datstr = fmt.format(v=ty[value].data)
            # replace scientific notation with something prettier
            if "e" in datstr:
                base, exponent = datstr.split("e")
                datstr = r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
            # write latex string to dict
            out[key + ab] = datstr
    # write all new commands to tex file
    gv.io.results_to_latex(out, file)
    return out


def load_ctd_profiles() -> xr.Dataset:
    """Load 2012 and 2014 CTD profiles.

    - Merge into one Dataset.
    - Calculate potential density referenced to 4000 dbar.

    Returns
    -------
    ctd : xr.Dataset
        CTD Dataset
    """
    cfg = load_config()
    ctd = xr.concat(
        [xr.open_dataset(file) for file in cfg.obs.input.ctd.glob("*.nc")], dim="n"
    )
    ctd = ctd.rename(dict(depth="z", n="x"))
    ctd = _calculate_sg4(ctd)
    return ctd


def downstream_ctd_profile():
    """Generate a far-field density profile from CTD data for towyo APE
    calculation and save to project data directory."""
    cfg = load_config()
    ctd = load_ctd_profiles()

    # Define downstream profiles
    ds = ctd.where(
        (ctd.lat > -7.9) & (ctd.lon > -169.5) & (ctd.lon < -168.5), drop=True
    )
    mlat_downstream = ds.lat.mean().data
    sg_downstream = ds.sg4.mean(dim="x")
    ztmp = -gsw.z_from_p(ctd.p.mean(dim="x"), mlat_downstream)
    intf = interp1d(ztmp, sg_downstream, bounds_error=False)
    z = np.arange(3000, 6001, 1)
    sg_downstream = intf(z)

    # Low pass filter profile
    FilterOrder = 3  # Filter order
    Wn = 1 / 200.0  # Cutoff frequency (this is 100m if sampled at 1m)
    B, A = sg.butter(FilterOrder, Wn, output="ba")
    sg4filt = np.ones_like(sg_downstream) * np.nan
    sg4filt[~np.isnan(sg_downstream)] = sg.filtfilt(
        B, A, sg_downstream[~np.isnan(sg_downstream)]
    )

    # Generate xarray.Dataset and save as netcdf
    ctd_ds = xr.Dataset(
        data_vars=dict(sigma_4=(("z"), sg_downstream), sg4filt=(("z"), sg4filt)),
        coords=dict(z=z),
    )
    ctd_ds.attrs["title"] = "downstream_density_ctd_profile"
    print("saving downstream CTD profile...")
    close_nc(cfg.obs.output.downstream_density_nc)
    ctd_ds.to_netcdf(cfg.obs.output.downstream_density_nc)


def _calculate_sg4(ds):
    r"""Calculate potential density anomaly referenced to 4000m ($\sigma_4$)
    from a CTD profile.

    Parameters
    ----------
    ds

    Returns
    -------
    ds
    """
    SA = gsw.SA_from_SP(ds.s, ds.p, ds.lon, ds.lat)
    sg4 = gsw.pot_rho_t_exact(SA, ds.t, ds.p, 4000) - 1000
    ds["sg4"] = (("z", "x"), sg4.data)
    return ds


def load_towyos(base=False):
    """Load towyos from 2012 and 2014 for analysis.

    Inputs
    ------
    base : bool
        Set to True to load only the data, not the netcdf files that also have
        a bunch of analysis ouput in addition. Defaults to False.

    Returns
    -------
    a : OrderedDict
        Dictionary with entries 't12' and 't14' for the two towyos.
    """
    config = load_config()

    if base:
        ty12 = xr.open_dataset(config.obs.output.towyo_2012_in)
        ty14 = xr.open_dataset(config.obs.output.towyo_2014_in)
    else:
        ty12 = xr.open_dataset(config.obs.output.towyo_2012)
        ty14 = xr.open_dataset(config.obs.output.towyo_2014)
    a = OrderedDict()
    a["t12"] = ty12
    a["t14"] = ty14

    return a


def close_nc(file, remove=True):
    """Close netcdf file (if existing).

    Workaround for replacing an existing & opened netcdf file.

    Parameters
    ----------
    file : pathlib.Path
        Path to netcdf file.
    remove : bool, optional
        If True (default), file will simply be deleted instead of closed.
    """
    if file.exists():
        if remove:
            file.unlink()
        else:
            print(f"closing nc file: {file.name}")
            tmp = xr.open_dataset(file)
            tmp.close()


def save_png(fname, subdir=None, **kwargs):
    """Save figure as png to the path defined in config.yml.

    Parameters
    ----------
    fname : str
        Figure name without file extension.
    """
    cfg = load_config()
    if subdir is not None:
        figdir = cfg.path.fig.joinpath(subdir)
    else:
        figdir = cfg.path.fig
    gv.plot.png(fname, figdir=figdir, **kwargs)


def save_pdf(fname, subdir=None):
    fname = fname + ".pdf"
    cfg = load_config()
    if subdir is not None:
        figdir = cfg.path.fig.joinpath(subdir)
        figdir.mkdir(exist_ok=True)
    else:
        figdir = cfg.path.fig
    print("saving pdf to {}/".format(figdir))
    plt.savefig(figdir.joinpath(fname), bbox_inches="tight", dpi=200)
