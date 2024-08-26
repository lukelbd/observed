#!/usr/bin/env python3
"""
Get scalar climate oscillation indices.
"""
# TODO: Revisit and merge with 'sst-bias' analysis utilities.
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from icecream import ic  # noqa: F401

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
# data is Kaplan SST and OISST). Finally NCEI seems to direct to same pages as CPC.
# See: https://psl.noaa.gov/data/climateindices/list/
# See: https://www.cpc.ncep.noaa.gov/data/indices/
# See: https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/teleconnections.shtml  # noqa: E501
# See: https://www1.ncdc.noaa.gov/pub/data/cmb/ersst/v5/index/
# See: https://crudata.uea.ac.uk/cru/data/pci.htm
# See: https://psl.noaa.gov/data/timeseries/AMO/
# See: https://psl.noaa.gov/gcos_wgsp/Timeseries/
# See: https://psl.noaa.gov/people/cathy.smith/best/
# See: https://psl.noaa.gov/enso/mei/
# See: https://climatedataguide.ucar.edu/climate-data/overview-climate-indices
INDEX_CODES = {
    'ao': 'nam',
    'aao': 'sam',
    'bei': 'best',
    'mei': 'enso',
    'nino34': 'nino',
    'nino3.4': 'nino',
}
INDEX_FILES = {
    'nino': ('ersst5.nino.mth.91-20.ascii.txt', 'nino34.long.anom.data.txt'),  # CPC ERSST vs. JPL HadSST  # noqa: E501
    'soi': ('soi.data.txt', 'soi.dat.txt'),  # JPL/CPC source vs. CRU source
    'pdo': ('ersst.v5.pdo.dat.txt', 'PDO.latest.txt'),  # CPC ERSST vs. JISAO HadSST/OISST  # noqa: E501
    'amo': ('ersst.v5.amo.dat.txt', 'amo_monthly.txt'),  # CPC ERSST vs. NCAR HadSST
    'nao': ('nao.data.txt', 'nao.dat.txt'),  # JPL/CPC source vs. CRU source
    'enso': 'meiv2.data.txt',  # PSL multivariate ENSO index (1979 to present)
    'best': 'enso.ts.1mn.txt',  # PSL bivariate ENSO index (1950 to present)
    'nam': 'monthly.ao.index.b50.current.ascii.table.txt',
    'sam': 'monthly.aao.index.b79.current.ascii.table.txt',
    'qbo': 'qbo.data.txt',
}
INDEX_LABELS = {
    'nino': 'Niño-3.4 Temperature',
    'soi': 'Southern Oscillation Index',
    'pdo': 'Pacific Decadal Oscillation',
    'amo': 'Atlantic Multidecadal Oscillation',
    'nao': 'North Atlantic Oscillation',
    'enso': 'Bivariate ENSO Index',
    'mei': 'Multivariate ENSO Index',
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
    codes = [INDEX_CODES.get(index, index) for index in indices]
    codes = codes or default
    datas, attrs = {}, {}
    for code, source in itertools.product(codes, sources):
        if cru is None and source == 'cru' and code not in tuples:
            continue
        kwargs.update(cru=source == 'cru')
        data = load_index(code, **kwargs)
        datas[code, source] = data
        attrs[code] = data.attrs
    data = pd.concat(datas.values(), keys=datas.keys(), axis='columns')
    data.columns.names = ('index', 'source')
    data.attrs = attrs
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
    code = INDEX_CODES.get(index, index)
    index_col = 0
    skipfooter = 0
    skiprows = 0  # default
    usecols = None  # default
    header = None  # default
    if code in ('nam', 'sam'):
        header = 0
        skipfooter = 0
    elif code in ('nao', 'soi'):
        skiprows, usecols = 1, range(13)  # shared
        skipfooter = 0 if cru else 3
    elif code == 'enso':
        skiprows, header, usecols = None, None, tuple(range(13))
        skipfooter = 0  # TODO: update
    elif code == 'best':
        skiprows, header = (1, None)
        skipfooter = 3
    elif code == 'amo':
        skiprows, header = (1, None) if cru else (1, 0)
        index_col = 0 if cru else (0, 1)
    elif code == 'pdo':
        skiprows, header = ((*range(30), 31), 0) if cru else (1, 0)
        skipfooter = 12 if cru else 0
    elif code == 'nino':
        skiprows, header = (1, None) if cru else (0, 0)
        skipfooter, index_col = (7, 0) if cru else (0, (0, 1))
    elif code == 'qbo':
        skiprows = 1
        skipfooter = 6
    else:
        raise ValueError(f'Invalid climate index {index!r}.')

    # Open data and convert to series
    # NOTE: Pandas colspecs='infer' seems to account for skiprows but not skipfooter,
    # so critical to reduce 'infer_nrows' from default of 100 for shorter datasets.
    options = INDEX_FILES[code]
    options = (options,) if isinstance(options, str) else options
    nulls = (9.99, 99.9, 99.99, 999, 999.9)
    if cru and len(options) == 1:
        raise ValueError(f'CRU version unavailable for climate index {code!r}.')
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
        if data.columns.size > 1 and code == 'nino':  # duplicates are e.g. 'anom.1'
            iloc = data.columns.str.lower().get_loc('nino3.4')  # Nino3.4 absolutes
            data = data.iloc[:, iloc + 1:iloc + 2]  # adjacent anomalies
        if data.columns.size > 1 or 'anom' not in data.columns[0].lower():
            raise RuntimeError(f'Error parsing {file!r}. Invalid columns {data.columns}.')  # noqa: E501
    else:
        if data.index.nlevels > 1 or data.columns.size != 12:
            raise RuntimeError(f'Error parsing {file!r}. Invalid shape {data.shape}.')
        if data.columns.is_integer():
            months = 1 + data.columns.values - data.columns.values.min()
        else:  # note strptime() seems case insensitive for month abbreviations
            months = [datetime.strptime(val, '%b').month for val in data.columns.values]
        data = data.set_axis(months, axis='columns')
        data = data.stack(dropna=False)  # WARNING: default behavior drops rows
    time = [f'{y:04d}-{m:02d}-01' for y, m in data.index]
    attrs = {}
    attrs['code'] = code.upper()
    attrs['units'] = ''
    attrs['long_name'] = INDEX_LABELS[code]  # climate index label
    attrs['short_name'] = ' '.join(INDEX_LABELS[code].split()[:-1])  # remove last word
    data = data.squeeze()  # convert to pandas series
    data.name = index  # standard name
    data.index = pd.DatetimeIndex(time, name='time')
    data.attrs = attrs
    return data
