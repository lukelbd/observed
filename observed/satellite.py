#!/usr/bin/env python3
"""
Load satellite-derived observations.
"""
import re
from pathlib import Path
from icecream import ic  # noqa: F401

import numpy as np  # noqa: F401
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401

from cmip_data.utils import assign_dates

__all__ = ['load_ceres', 'load_erbe']


def load_ceres(path=None, path0=None, *, average=True, anomaly=True, **kwargs):
    """
    Return a dataset with CERES EBF radiative flux estimates.

    Parameters
    ----------
    path : path-like
        The source file or directory.
    path0 : path-like
        The climate file or directory.
    average : bool, optional
        Whether to load global data.
    anomaly : bool, optional
        Whether to compute anomalies.
    **kwargs
        Passed to `xarray.Dataset.sel`.
    """
    # NOTE: Standardized data produced in 'climate-data/process.py'
    # NOTE: Full 23 year data uses only TERRA from 2000-2003, TERRA + AQUA from
    # 2003-2019, and NOAA20 afterwards. Product includes bias and drift corrections
    # to prevent discontinuity when satellites change (however could try to detect).
    from .datasets import _parse_path
    folder = Path('~/data/ceres').expanduser()
    iglob = '[0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]'
    name = 'global' if average else 'standardized'
    glob0 = f'CERES_EBAF-TOA_*_CLIM01-CLIM12_{name}.nc'
    glob = f'CERES_EBAF-TOA_*_{iglob}_{name}.nc'
    paths = (path0, path) if anomaly else (path,)
    globs = (glob0, glob) if anomaly else (glob,)
    regex = re.compile(r'\Ag?([ts])[^_]*_([lsn])[^_]*_((?=a)|clr).*\Z')
    datas = []
    for path, glob in zip(paths, globs):
        path = _parse_path(path, folder, glob, search=True)
        data = xr.open_dataset(path, engine='netcdf4', use_cftime=True)
        rename = {
            key: 'clt' if 'cldarea' in key
            else 'tau' if 'cldtau' in key
            else 'ttop' if 'cldtemp' in key
            else 'ptop' if 'cldpress' in key
            else 'rsdt' if 'solar' in key
            else regex.sub(r'r\2n\1\3', key)
            .replace('nn', 'fn')  # 'net' to 'full'
            .replace('clr', 'cs')  # 'clear-sky' suffix
            .replace('rsn', 'rsu')  # shortwave variables are outgoing
            .replace('rln', 'rlu')  # longwave variables are upwelling
            for key in data.data_vars
        }
        data = data.rename(rename)
        if 'ctime' in data.sizes and np.all(data.ctime == 0):
            data = data.assign_coords(ctime=np.arange(1, 13))  # cdo standardize bug
        if 'ctime' in data.sizes:
            data = assign_dates(data.rename(ctime='month'), year=1800)
        else:  # assign dates but keep years
            data = assign_dates(data)
        for array in data.data_vars.values():
            if 'long_name' in array.attrs:  # dictionary insertion order
                array.attrs['long_name'] = array.attrs.pop('long_name')
            if 'standard_name' in array.attrs:  # backwards for some reason
                array.attrs['short_name'] = array.attrs.pop('standard_name')
            if 'CF_name' in array.attrs:
                array.attrs['standard_name'] = array.attrs.pop('CF_name')
        datas.append(data)
    if len(datas) == 1:  # original data
        data, = datas
    else:  # relative to climatology
        clim = datas[0].groupby('time.month').mean()  # no-op
        with xr.set_options(keep_attrs=True):
            data = datas[1].groupby('time.month') - clim
        data = data.drop_vars('month')
    return data.sel(**kwargs)


def load_erbe(path=None):
    """
    Return a dataset with ERBE radiative flux estimates.

    Parameters
    ----------
    base, file : path-like
        The directory and file name.
    clim : path-like
        The climatology file name.
    average : bool, optional
        Whether to load global data.
    anomaly : bool, optional
        Whether to compute anomalies.
    **kwargs
        Passed to `xarray.Dataset.sel`.
    """
    # NOTE: Standardized data produced in 'climate-data/process.py'
    path  # TODO: Complete this
