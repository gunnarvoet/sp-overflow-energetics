[![made-with-datalad](https://www.datalad.org/badges/made_with.svg)](https://datalad.org)

Energy and Momentum of a Density-Driven Overflow in the Samoan Passage
======================================================================

A study of energetics and momentum of the Samoan Passage Northern Sill overflow based on towyo observations and a two-dimensional MITgcm model run.

Observational datasets from the [Samoan Passage Data Archive](https://github.com/gunnarvoet/sp-data-archive) have been installed into [data/in/](data/in/) via [DataLad](https://www.datalad.org/) / [git-annex](https://git-annex.branchable.com/). The Makefile contains the necessary steps to retrieve the data. To make use of this functionality, this repository must be cloned using DataLad, more on this [below](#datalad-datasets-and-how-to-use-them). Alternatively, data can be manually downloaded from the data archive.

Model data are stored on google drive and can be retrieved via `make` calls.

The source code to run all analysis steps and generate plots for the manuscript is contained in [code/](code/). All analysis steps are documented in the [Makefile](Makefile), type `make help` for an overview.

The source code for the manuscript is contained in [doc/](doc/).

Analysis code is shared under the [MIT License](https://opensource.org/licenses/MIT). Manuscript code and figures are shared under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

----------

### DataLad datasets and how to use them

This repository is a [DataLad](https://www.datalad.org/) dataset. It provides
fine-grained data access down to the level of individual files, and allows for
tracking future updates. In order to use this repository for data retrieval,
[DataLad](https://www.datalad.org/) is required. It is a free and
open source command line tool, available for all major operating
systems, and builds up on Git and [git-annex](https://git-annex.branchable.com/)
to allow sharing, synchronizing, and version controlling collections of
large files. You can find information on how to install DataLad at
[handbook.datalad.org/en/latest/intro/installation.html](http://handbook.datalad.org/en/latest/intro/installation.html).

#### Get the dataset

A DataLad dataset can be `cloned` by running

```
datalad clone <url>
```

Once a dataset is cloned, it is a light-weight directory on your local machine.
At this point, it contains only small metadata and information on the
identity of the files in the dataset, but not actual *content* of the
(sometimes large) data files.

#### Retrieve dataset content

After cloning a dataset, you can retrieve file contents by running

```
datalad get <path/to/directory/or/file>`
```

This command will trigger a download of the files, directories, or
subdatasets you have specified.

DataLad datasets can contain other datasets, so called *subdatasets*.
If you clone the top-level dataset, subdatasets do not yet contain
metadata and information on the identity of files, but appear to be
empty directories. In order to retrieve file availability metadata in
subdatasets, use `-n` flag like so:

```
datalad get -n <path/to/subdataset>
```

Afterwards, you can browse the retrieved metadata to find out about
subdataset contents, and use `datalad get` once again (no flag this time) to retrieve individual files.
If you use `datalad get <path/to/subdataset>`, all contents of the
subdataset will be downloaded at once.

#### Stay up-to-date

DataLad datasets can be updated. The command `datalad update` will
*fetch* updates and store them on a different branch (by default
`remotes/origin/master`). Running

```
datalad update --merge
```

will *pull* available updates and integrate them in one go.

#### Find out what has been done

DataLad datasets contain their history in the ``git log``.
By running ``git log`` (or a tool that displays Git history) in the dataset or on
specific files, you can find out what has been done to the dataset or to individual files
by whom, and when.

#### More information

More information on DataLad and how to use it can be found in the DataLad Handbook at
[handbook.datalad.org](http://handbook.datalad.org/en/latest/index.html). The chapter
"DataLad datasets" can help you to familiarize yourself with the concept of a dataset.

