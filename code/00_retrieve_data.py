import argparse
import shutil
import tarfile

from datalad.api import get as dl_get
from datalad.api import drop as dl_drop
import gdown

import nslib as nsl


def download_model_extracted():
    cfg = nsl.io.load_config()
    output_dir = cfg.path.output
    # model data for analysis time period
    model_extracted = cfg.parameters.google_drive_ids.model_extracted
    id = model_extracted.id
    out = output_dir.joinpath(model_extracted.file)
    if out.exists() is False:
        gdown.download(id=id, output=out.as_posix(), quiet=False)
    # sorted initial density field
    model_sorted_initial_density = (
        cfg.parameters.google_drive_ids.model_sorted_initial_density
    )
    id = model_sorted_initial_density.id
    out = output_dir.joinpath(model_sorted_initial_density.file)
    if out.exists() is False:
        gdown.download(id=id, output=out.as_posix(), quiet=False)
    # sorted reference density profile
    model_refrho_sorted = (
        cfg.parameters.google_drive_ids.model_refrho_sorted
    )
    id = model_refrho_sorted.id
    out = output_dir.joinpath(model_refrho_sorted.file)
    if out.exists() is False:
        gdown.download(id=id, output=out.as_posix(), quiet=False)
    # reference density profile
    model_refrho = (
        cfg.parameters.google_drive_ids.model_refrho
    )
    id = model_refrho.id
    out = output_dir.joinpath(model_refrho.file)
    if out.exists() is False:
        gdown.download(id=id, output=out.as_posix(), quiet=False)


def download_model_raw():
    cfg = nsl.io.load_config()
    id = cfg.parameters.google_drive_ids.model_full.id
    out_dir = cfg.model.input.full_model_run
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_archive = out_dir.joinpath(cfg.parameters.google_drive_ids.model_full.file)
    gdown.download(id=id, output=tar_archive.as_posix(), quiet=False)
    # extract tar archive
    tar = tarfile.open(tar_archive)
    tar.extractall(path=out_dir)
    tar.close()
    # remove tar file
    tar_archive.unlink()


def clean_model_raw():
    shutil.rmtree(cfg.model.input.full_model_run, ignore_errors=True)


def get_bathy():
    cfg = nsl.io.load_config()
    directory = cfg.obs.input.bathy.parent
    dl_get(directory)


def drop_bathy():
    cfg = nsl.io.load_config()
    directory = cfg.obs.input.bathy.parent
    dl_drop(directory)


def get_ctd():
    cfg = nsl.io.load_config()
    directory = cfg.obs.input.ctd
    dl_get(directory)


def drop_ctd():
    cfg = nsl.io.load_config()
    directory = cfg.obs.input.ctd
    dl_drop(directory)


def get_mp():
    cfg = nsl.io.load_config()
    mp_file = cfg.obs.input.mp_t1
    dl_get(mp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve model data stored on Google Drive."
    )
    parser.add_argument(
        "--bathy", help="retrieve bathymetric data", action="store_true"
    )
    parser.add_argument(
        "--mp", help="retrieve MP data", action="store_true"
    )
    parser.add_argument(
        "--ctd", help="retrieve stationary CTD/LADCP data", action="store_true"
    )
    parser.add_argument("--towyo", help="retrieve towyo data", action="store_true")
    parser.add_argument(
        "--model_raw", help="retrieve raw model output", action="store_true"
    )
    parser.add_argument(
        "--model_extracted",
        help="retrieve model output for analysis period in netcdf format",
        action="store_true",
    )
    parser.add_argument(
        "--drop", help="will drop instead of get data", action="store_true"
    )

    args = parser.parse_args()

    if args.bathy:
        if args.drop:
            drop_bathy()
        else:
            get_bathy()

    if args.ctd:
        if args.drop:
            drop_ctd()
        else:
            get_ctd()

    if args.towyo:
        get_towyo()

    if args.mp:
        get_mp()

    if args.model_extracted:
        download_model_extracted()

    if args.model_raw:
        download_model_raw()
