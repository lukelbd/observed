#!/usr/bin/env python3
"""
Load gridded surface observations.
"""
from datetime import datetime
from pathlib import Path
from icecream import ic  # noqa: F401

import cftime
import numpy as np  # noqa: F401
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401

from cmip_data.utils import assign_dates

__all__ = ['load_gistemp', 'load_hadcrut']


def load_hadcrut(path=None, average=True, **kwargs):
    """
    Return a dataset with HadCRUT surface temperature estimates.

    Parameters
    ----------
    path : path-like
        The directory or file name.
    average : bool, optional
        Whether to load global averages.
    **kwargs
        Passed to `xarray.Dataset.sel`.
    """
    # TODO: Incorporate observational uncertainty not just regression uncertainty?
    # NOTE: Standardized data produced in 'climate-data/process.py'
    # NOTE: Jones 'Apples and Oranges' paper found that comparing HadCRUT with
    # model surface temperature should has only minor effects on results.
    from .datasets import _parse_path
    folder = Path('~/data/hadcrut5').expanduser()
    name = 'summary_series' if average else 'anomalies'
    mode = 'global.monthly' if average else 'ensemble_mean_standardized'
    file = f'HadCRUT.*.analysis.{name}.{mode}.nc'
    path = _parse_path(path, folder, file, search=True)
    data = xr.open_dataset(path, use_cftime=True)
    rename = dict(latitude='lat', longitude='lon')  # cdo will rename
    rename.update(tas_mean='ts', tas_lower='ts_lower', tas_upper='ts_upper')
    rename = {key: value for key, value in rename.items() if key in data}
    data = data.rename(rename)
    drop = ('time_bnds', 'longitude_bnds', 'latitude_bnds', 'realization_bnds')
    drop = data.data_vars.keys() & set(drop)
    drop = drop | (data.coords.keys() - data.sizes.keys())
    data = data.drop(drop)
    data = assign_dates(data)
    return data.sel(**kwargs)


def load_gistemp(path=None, average=True, **kwargs):
    """
    Return a dataset with GISTEMP surface temperature estimates.

    Parameters
    ----------
    path : path-like
        The directory or file name.
    average : bool, optional
        Whether to load global data.
    **kwargs
        Passed to `xarray.Dataset.sel`.
    """
    # NOTE: Standardized data produced in 'climate-data/process.py'
    # NOTE: This is adapted from utils.assign_dates(). Should not bother
    # implementing there since this is special case.
    from .datasets import _parse_path
    folder = Path('~/data/gistemp4').expanduser()
    suffix = 'global.csv' if average else 'standardized.nc'
    file = f'gistemp1200_GHCNv?_ERSSTv?_{suffix}'
    path = _parse_path(path, folder, file, search=True)
    if path.suffix == '.nc':
        data = xr.open_dataset(path, use_cftime=True)
        data = data.rename(tempanomaly='ts')
        data = data.drop_vars('time_bnds')
    else:
        months = xr.DataArray(np.arange(1, 13), dims='month', name='month')
        usecol = [datetime(2000, month, 1).strftime('%b') for month in months.values]
        data = pd.read_csv(path, header=1, index_col='Year', na_values='***', usecols=['Year', *usecol])  # noqa: E501
        data = data.to_xarray().rename(Year='year')
        data = xr.concat(data.data_vars.values(), months)
        data = data.to_dataset(name='ts').stack(time=('year', 'month'))
        time = [cftime.datetime(y, m, 15) for y, m in zip(data.year.values, data.month.values)]  # noqa: E501
        time = xr.DataArray(xr.CFTimeIndex(time), dims='time')
        data = data.drop_vars(data.keys() & {'month', 'year'})
        data = data.assign_coords(time=time)
        data.ts.attrs.update({'units': 'K', 'long_name': 'surface temperature'})
    return data.sel(**kwargs)
