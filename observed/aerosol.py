#!/usr/bin/env python3
"""
Functions for loading aerosol data.
"""
# TODO: Add AERmon table 'od550aer' non-athropogenic aerosol optical depth, input4MIPs
# historical 'SO2' non-anthropogenic emissions, or CCMI volcanic aerosol dataset.
# See: https://blogs.reading.ac.uk/ccmi/ccmi-2022_archive/
# See: https://gmd.copernicus.org/articles/9/4049/2016/
# See: https://cds.climate.copernicus.eu/cdsapp#!/dataset/app-climate-monitoring-volcanoes?tab=overview  # noqa: E501
import numpy as np
import xarray as xr
from icecream import ic  # noqa: F401

from .arrays import annual_average

__all__ = ['load_volcanic']


def load_volcanic(path=None):
    """
    Return Ammann (2003) volcanic aerosol optical depth.

    Parameters
    ----------
    path : path-like, optional
        The emissions data path.
    """
    # TODO: Support preserving 'time' coordinate instead of converting to year.
    # NOTE: This dataset captures all major eruptions since post-Pinatubo
    # has been quiescent (compare with Figure 1 from VolImpact paper below).
    # See: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2003GL016875
    # See: https://doi.org/10.1127/metz/2019/0999
    path = path or '~/data/volcanic/ammann2003b_volcanics.nc'
    data = xr.open_dataarray(path)
    data.name = 'tau_strat'  # stratospheric optical depth
    data.attrs.update(units='')
    time = data.time.astype(str).values  # bizarre time format
    time = [np.datetime64(f'{t[:4]}-{t[4:]}-01') for t in time]
    data = data.climo.replace_coords(time=np.array(time))
    # data = data.climo.add_scalar_coords()
    # data = data.climo.add_cell_measures(('width', 'depth'))  # TODO: retore this
    # return annual_average(data.climo.average('lat'))
    wgts = np.cos(data.climo.coords.get('lat', quantify=True))
    data = (data * wgts).sum('lat') / wgts.sum('lat')
    data = annual_average(data)
    data = data.rename(time='year')
    data = data.assign_coords(year=data.year.dt.year)
    return data
