#!/usr/bin/env python3
"""
Get scalar climate oscillation indices.
"""
# TODO: Revisit and merge with 'sst-bias' analysis utilities.
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic  # noqa: E501

__all__ = ['load_index', 'load_indices']


# Data source links
# NOTE: Processes versions of indices e.g. 'ONI' (3-month running mean ERSST5 Nino 3.4
# used operationally by NOAA) or 'centered ENSO' (and 10-year low-pass-filter but
# instead support this in regression functions. Use climopy lagged regression and
# spectral filtering tools and subtract monthly 1981-2010 or 1991-2020 climatologies
# or even cliamte drift-corrected climatologies to get proper index anomalies.
# NOTE: Raw data is from CPC and CRU while PSL provides links and irregularly-updated
# copies. Exceptions are regularly-updated HadISST ENSO indices, not available from CRU
# site, and regularly-updated CPC/ERSST NAO indices, not available from CPC site. Also
# NCAR page mostly links to externally hosted data but provides custom HadISST based
# AMO (not available from CRU or PSL) and custom SOI versions (but this is available
# from CRU and PSL so skip). Note the PSL 'gcos_wgsp' page has both PSL/CRU-derived
# HadISST indices while 'climateindices' has CPC-derived ERSST indices (but for AMO
# ERSST data have to go to general web page which points to NCDC source -- the default
# data is Kaplan SST and OISST). Finally NCEI seems to direct to same pages as NCEI.
# See: https://www.cpc.ncep.noaa.gov/data/indices/
# See: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/teleconnections.shtml  # noqa: E501
# See: https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/index/
# See: https://crudata.uea.ac.uk/cru/data/pci.htm
# See: https://psl.noaa.gov/data/climateindices/list/
# See: https://psl.noaa.gov/data/timeseries/AMO/
# See: https://psl.noaa.gov/gcos_wgsp/Timeseries/
# See: https://climatedataguide.ucar.edu/climate-data/overview-climate-indices
INDEX_CODES = {'ao': 'nam', 'aao': 'sam', 'nino34': 'nino'}
INDEX_FILES = {
    'nino': ('ersst5.nino.mth.91-20.ascii.txt', 'nino34.long.anom.data.txt'),  # CPC ERSST vs. JPL HadSST  # noqa: E501
    'soi': ('soi.data.txt', 'soi.dat.txt'),  # JPL/CPC source vs. CRU source
    'pdo': ('ersst.v5.pdo.dat.txt', 'PDO.latest.txt'),  # CPC ERSST vs. JISAO HadSST/OISST  # noqa: E501
    'amo': ('ersst.v5.amo.dat.txt', 'amo_monthly.txt'),  # CPC ERSST vs. NCAR HadSST
    'nao': ('nao.data.txt', 'nao.dat.txt'),  # JPL/CPC source vs. CRU source
    'nam': 'monthly.ao.index.b50.current.ascii.table.txt',
    'sam': 'monthly.aao.index.b79.current.ascii.table.txt',
    'qbo': 'qbo.data.txt',
}
INDEX_LABELS = {
    'nino': 'Niño-3.4 Temperature Index',
    'soi': 'Southern Oscillation Index',
    'pdo': 'Pacific Decadal Oscillation',
    'amo': 'Atlantic Multidecadal Oscillation',
    'nao': 'North Atlantic Oscillation',
    'nam': 'Northern Annular Mode',  # or Arctic Oscillation
    'sam': 'Southern Annular Mode',  # or Antarctic Oscillation
    'qbo': 'Quasi-Biennial Oscillation',
}


def load_indices(*indices, cru=None, **kwargs):
    """
    Load input monthly climate indices.

    Parameters
    ----------
    *indices : str, optional
        The index names. Default depends on inputs.
    cru : bool, optional
        Used to determine default indices.
    **kwargs
        Passed to `load_index`.
    """
    # NOTE: This loads both ESRL and CRU data sources at once. In future should
    # do same for all available sources of CO2 station data.
    tuples = [index for index, file in INDEX_FILES.items() if not isinstance(file, str)]
    default = tuples if cru else INDEX_FILES
    sources = ('cpc', 'cru') if cru is None else ('cru',) if cru else ('cpc',)
    indices = [INDEX_CODES.get(index, index) for index in indices]
    indices = indices or default
    datas = {}
    for index in indices:
        for source in sources:
            if cru is None and source == 'cru' and index not in tuples:
                continue
            kwargs.update(cru=source == 'cru')
            data = load_index(index, **kwargs)
            datas[(index, source)] = data
    data = pd.concat(datas.values(), keys=datas.keys(), axis='columns')
    data.columns.names = ('index', 'source')
    return data


def load_index(index, base=None, cru=False):
    """
    Load input monthly climate index.

    Parameters
    ----------
    index : str
        The index to load.
    base : path-like
        The base directory path.
    cru : bool, optional
        Whether to use data from the Met Office CRU instead of NOAA CPC. This will
        raise an error if set to ``True`` and only NOAA CPC data is available.
    """
    # File settings
    # NOTE: See above for details. Generally have either CRU or CPC versions that use
    # either HadSSST or ERSST. The non-SST indices are also different between versions.
    base = Path(base or '~/data/indices').expanduser()
    index = INDEX_CODES.get(index, index)
    index_col = 0
    skipfooter = 0
    skiprows = 0  # default
    usecols = None  # default
    header = None  # default
    if index in ('nam', 'sam'):
        header = 0
        skipfooter = 0
    elif index in ('nao', 'soi'):
        skiprows, usecols = 1, range(13)  # shared
        skipfooter = 0 if cru else 3
    elif index == 'amo':
        skiprows, header = (1, None) if cru else (1, 0)
        index_col = 0 if cru else (0, 1)
    elif index == 'pdo':
        skiprows, header = ((*range(30), 31), 0) if cru else (1, 0)
        skipfooter = 12 if cru else 0
    elif index == 'nino':
        skiprows, header = (1, None) if cru else (0, 0)
        skipfooter, index_col = (7, 0) if cru else (0, (0, 1))
    elif index == 'qbo':
        skiprows = 1
        skipfooter = 6
    else:
        raise ValueError(f'Invalid climate index {index!r}.')

    # Open data and convert to series
    # NOTE: Pandas colspecs='infer' seems to account for skiprows but not skipfooter,
    # so critical to reduce 'infer_nrows' from default of 100 for shorter datasets.
    options = INDEX_FILES[index]
    options = (options,) if isinstance(options, str) else options
    nulls = (99.9, 99.99, 999, 999.9)
    if cru and len(options) == 1:
        raise ValueError(f'CRU version unavailable for climate index {index!r}.')
    file = options[int(bool(cru))]
    data = pd.read_fwf(
        base / file,
        header=header,  # header row number (starts at zero)
        skiprows=skiprows,  # number of lines ('header' is relative to this)
        skipfooter=skipfooter,  # number of lines (requires 'python' engine)
        skip_blank_lines=False,  # skip manually for consistency
        index_col=index_col,  # 'year' column or 'year-month' columns
        infer_nrows=50,  # ignore footer text and full-width -999.99 fill values
        usecols=usecols,  # skip e.g. final annual-average column
        engine='python',
        dtype=np.float64,
    )
    for null in nulls:  # see files for exact values used
        mask = np.isclose(np.abs(data.values), null)
        data.values[mask] = np.nan
    if data.index.nlevels == 2:
        if data.columns.size == 1:  # enforce name for check at the bottom
            data.columns = ('anom',)
        if data.columns.size > 1 and index == 'nino':  # duplicates are e.g. 'anom.1'
            loc = data.columns.str.lower().get_loc('nino3.4')  # Nino3.4 absolutes
            data = data.iloc[:, loc + 1:loc + 2]  # adjacent anomalies
        if data.columns.size > 1 or 'anom' not in data.columns[0].lower():
            raise RuntimeError(f'Error parsing {file!r}. Invalid columns {data.columns}.')  # noqa: E501
    else:
        if data.index.nlevels > 1 or data.columns.size != 12:
            raise RuntimeError(f'Error parsing {file!r}. Invalid shape {data.shape}.')
        if data.columns.is_integer():
            months = 1 + data.columns.values - data.columns.values.min()
        else:  # note strptime() seems case insensitive for month abbreviations
            months = [datetime.strptime(val, '%b').month for val in data.columns.values]
        data = data.set_axis(months, axis='columns').stack()
    time = [f'{y:04d}-{m:02d}-01' for y, m in data.index]
    data = data.squeeze()  # convert to pandas series
    data.name = index
    data.index = pd.DatetimeIndex(time, name='time')  # noqa: E501
    return data
