#!/usr/bin/env python3
"""
Open datasets containing multiple observed variables.
"""
from pathlib import Path

import xarray as xr
from cmip_data import assign_dates

from .satellite import load_ceres
from .surface import load_gistemp, load_hadcrut

__all__ = ['load_dataset', 'load_shared']


def load_dataset(
    base=None, ceres=None, ceres0=None, gistemp=None, hadcrut=None, globe=True,
):
    """
    Return a dataset with radiative flux and surface temperature observations.

    Parameters
    ----------
    base : path-like, optional
        The directory name.
    ceres, ceres0 : path-like, optional
        The CERES file and climatology name.
    gistemp, hadcrut : path-like, optional
        The GISTEMP4 and HadCRUT5 file name.
    globe : bool, optional
        Whether to load global data.
    """
    ceres = load_ceres(base, ceres, clim=ceres0, globe=globe)
    slice_ = slice(ceres.time.values[0], ceres.time.values[-1])
    rename = dict(ts='ts_gis')  # rename single variable
    gistemp = load_gistemp(base, gistemp, globe=globe, time=slice_)
    gistemp = gistemp.rename(rename)
    rename = dict(ts='ts_had', ts_lower='ts_had_lower', ts_upper='ts_had_upper')
    hadcrut = load_hadcrut(base, hadcrut, globe=globe, time=slice_)
    hadcrut = hadcrut.rename(rename if globe else dict(ts='ts_had'))
    datasets = {'ceres': ceres, 'gistemp': gistemp, 'hadcrut': hadcrut}  # noqa: E501
    if globe:  # He et al. shared data
        rename = dict(ts='ts_he')  # preserve cloud names
        shared = load_shared(base, time=slice_)
        datasets['shared'] = shared.rename(rename)
    sizes = {
        src: 'x'.join(str(data.sizes.get(key, 1)) for key in ('lon', 'lat'))
        for src, data in datasets.items()  # print coordinate sizes
    }
    if len(set(sizes.values())) != 1:  # used before data were standardized
        message = ', '.join(f'{src} {size}' for src, size in sizes.items())
        raise RuntimeError(f'Incompatible data dimensions: {message}')
    dataset = xr.merge(datasets.values())
    return dataset


def load_shared(base=None, ceres=None, gistemp=None, **kwargs):
    """
    Return a dataset with cloud response and surface temperature from Haozhe He.

    Parameters
    ----------
    base : path-like
        The source directory.
    ceres : optional
        The source radiative flux file.
    gistemp : optional
        The source surface temperature file.
    **kwargs
        Passed to `xarray.Dataset.sel`.
    """
    # NOTE: Reproduce He et al. results by skipping detrending when computing
    # feedbacks. Otherwise uncertainty much higher.
    # TODO: Here use decode_coords='all' to bypass 'time_bnds' variable that
    # otherwise prohibits xr.open_dataarray(). Should use this elsewhere.
    base = Path(base or '~/data/shared').expanduser()
    flux = ceres or 'He_CERES_TOA-cloud_200101-201912_global-anom.nc'
    flux = xr.open_dataset(base / flux, use_cftime=True)
    flux = assign_dates(flux)
    flux = flux.squeeze(drop=True)  # remove lon and lat
    for name in flux:
        cloud = 'cld' if 'cloud' in name else ''
        part = 'hi' if 'high' in name else 'lo' if 'low' in name else 'mx' if 'mixed' in name else ''  # noqa: E501
        rad = 'rlnt' if 'LW' in name else 'rsnt' if 'SW' in name else 'rfnt'
        flux = flux.rename({name: f'{cloud}{part}_{rad}'.strip('_')})
    temp = gistemp or 'He_gistemp1200_GHCNv4_ERSSTv5_200101-201912_global-anom.nc'
    temp = xr.open_dataarray(base / temp, use_cftime=True, decode_coords='all')
    temp = assign_dates(temp)
    temp.name = 'ts'  # change from 'tempanomaly'
    data = xr.merge((temp, flux))  # note both are from 2001 to 2019
    data = data.sel(**kwargs)
    return data
