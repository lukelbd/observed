#!/usr/bin/env python3
"""
Load individual station observations.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from climopy import const, ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

__all__ = ['load_station', 'load_stations']


# Station names
# These are for internal use
STATION_NAMES = {
    'gl': 'globe',
    'alt': 'alert',
    'ptb': 'barrow',
    'ljo': 'la-jolla',
    'mlo': 'mauna-loa',
    'kum': 'kumukahi',
    'fan': 'fanning',
    'chr': 'christmas',
    'sam': 'samoa',
    'ker': 'kermadec',
    'nzd': 'baring',
    'spo': 'south-pole',
}
STATION_CODES = {name: code for code, name in STATION_NAMES.items()}

# Station labels
# See: https://scrippsco2.ucsd.edu/data/atmospheric_co2/sampling_stations.html
STATION_LABELS = {
    'gl': 'global average',
    'alt': 'Alert, NWT, Canada',
    'ptb': 'Barrow, Alaska',
    'ljo': 'La Jolla Pier, California',
    'mlo': 'Mauna Loa Observatory, Hawaii',
    'kum': 'Cape Kumukahi, Hawaii',
    'fan': 'Fanning Island',
    'chr': 'Christmas Island',
    'sam': 'American Samoa',
    'ker': 'Kermadec Island',
    'nzd': 'Baring Head, New Zealand',
    'spo': 'South Pole',
}

# Station metadata
# See: https://scrippsco2.ucsd.edu/data/atmospheric_co2/sampling_stations.html
STATION_COORDS = {  # (t, z, y, x) i.e. (first year, elevation, latitude, longitude)
    'gl': (1979, None, None, None),  # see 'gl' esrl csv file
    'alt': (1985, 210, 82.3, -62.3),
    'ptb': (1961, 11, 71.3, -156.6),
    'ljo': (1957, 10, 32.9, -117.3),
    'mlo': (1958, 3397, 19.5, -155.6),  # both esrl and scripps
    'kum': (1979, 3, 19.5, -154.8),
    'fan': (1977, 3, 3.9, -159.4),
    'chr': (1974, 2, 2.0, -157.3),
    'sam': (1981, 30, -14.2, -170.6),
    'ker': (1982, 2, -29.2, -177.9),
    'nzd': (1977, 85, -41.4, 174.9),
    'spo': (1957, 2810, -90.0, 0.0),
}


def load_stations(
    *stations, annual=None, growth=None, daily=None, esrl=None, **kwargs,
):
    """
    Load CO$_2$ oberving stations.

    Parameters
    ----------
    *stations : str, optional
        The station names. Default depends on inputs.
    annual, growth, daily, esrl : optional
        Used to determine the default stations.
    **kwargs
        Passed to `load_station`.
    """
    # TODO: Support loading multiple data sources at once. Can be used to get rough
    # uncertainy estimate and for robustness test. For now suspect climate indices and
    # observational data have more uncertainty so leave this alone.
    # NOTE: This changes default stations depending on user input. When reqeusting
    # ESRL data only load global average and Mauna Loa. By default will return ESRL
    # global average estimate along with Scripps individual station data.
    stations = [STATION_CODES.get(station, station) for station in stations]
    annual = not daily if daily is not None else growth if annual is None else annual
    esrl = annual if esrl is None else esrl
    datas, units = {}, None
    kwargs.update(annual=annual, daily=daily, growth=growth, esrl=esrl)
    if stations:
        sorted_ = [code for code in STATION_NAMES if code in stations]
        codes = sorted_ + [code for code in stations if code not in sorted_]
    else:
        stations = [name for name in STATION_NAMES if name != 'gl']
        codes = stations if esrl == 0 else ('gl', 'mlo') if esrl == 1 else STATION_NAMES
    for code in codes:
        data = load_station(code, **kwargs)
        name = data.name  # station name
        if units is None:
            data, units = data.magnitude, data.units
        else:
            data = data.to(units).magnitude
        datas[name] = data
    data = pd.concat(datas.values(), keys=datas.keys(), axis='columns')
    data = ureg.Quantity(data, units)
    return data


def load_station(
    station=None,
    base=None,
    annual=None,
    growth=False,
    daily=False,
    decimal=False,
    flasks=None,
    esrl=None,
    raw=False,
):
    """
    Load CO$_2$ concentration station data.

    Parameters
    ----------
    station : str, optional
        The station name or code. Default is Mauna Loa.
    base : path-like
        The directory name.
    annual : bool, optional
        Whether to load annual data instead of monthly. Default is ``growth``.
    growth : bool, optionl
        Whether to load annual growth rates instead of raw concentrations.
    daily : bool, optional
        Whether to load daily data instead of monthly. Default is ``False``.
    decimal : bool, optional
        Whether to use the decimal year index instead of datetime index.
    flasks : bool, optional
        Whether to load flask data. Default is to prefer merged then in situ then flask.
    esrl : bool, optional
        Whether to load Scripps data. Default is ``True`` except for global average.
    raw : bool, optional
        Whether to load raw data instead of interpolated data. Not availble for ESRL.

    Returns
    -------
    data : pandas.DataFrame
        The station data.
    """
    # Station settings
    # TODO: Switch to pint-pandas support for quantifying underlying data.
    # NOTE: Scripps and ESLR have generally consistent format but some stations
    # still have exceptions. Should add to the below when reading new sources.
    station = station or 'mlo'
    base = Path(base or '~/data/stations-co2').expanduser()
    code = STATION_CODES.get(station, station)
    if code not in STATION_NAMES:
        raise ValueError(f'Invalid station name or code {station!r}.')
    names_idx, names_data, names_use = ['year', 'month', 'day'], None, None
    annual = growth if annual is None else annual and not daily
    esrl = annual or code == 'gl' if esrl is None else esrl
    if esrl:  # esrl data
        sep = ','  # standard csv
        freq = 'gr' if growth else 'annmean' if annual else 'daily' if daily else 'mm'
        paths = (base / f'co2_{freq}_{code}.csv',)
        comment = '#'  # comment character
        if raw:  # un-interpolated
            raise ValueError('ESRL only provides interpolated time series.')
        if not paths[0].is_file():
            message = f'{paths[0]!s}'  # see below
            raise FileNotFoundError(f'ESRL {code} file not found: {message}')
        if daily:  # note only global data has de-seasonalized trend
            header = 0 if code == 'gl' else None  # header commented out
            names_idx = names_idx if code == 'gl' else names_idx + ['decimal']
            names_data = ['mean', 'trend'] if code == 'gl' else ['mean']
        else:  # monthly or annual data
            header = 0  # header row
            names_ml = ['mean', 'trend', 'days', 'days sigma', 'mean sigma']
            names_gl = ['mean', 'mean sigma', 'trend', 'trend sigma']
            names_ann = ['mean', 'mean sigma']  # two column annual data
            names_idx = ['year'] if annual else ['year', 'month', 'decimal']
            names_data = names_ann if annual else names_gl if code == 'gl' else names_ml
    else:  # scripps data
        in_situ_mlo = not flasks and code == 'mlo'  # WARNING: strange special case
        sep = r',\s*'  # WARNING: mlo files require appending ',' to header columns
        freq = 'daily' if daily else 'monthly'
        opts = ('merge', 'in_situ', 'flask')
        opts = opts if flasks is None else ('flask',) if flasks else ('in_situ',)
        files = [f'{freq}_{opt}_co2_{code}.csv' for opt in opts]
        paths = [base / file for file in files if (base / file).is_file()]
        comment = '%' if daily and in_situ_mlo else '"'  # special case
        if growth or annual:
            raise ValueError('Scripps does not provide annual data or growth rates.')
        if not paths:
            message = ', '.join(f'{base / file!s}' for file in files)
            raise FileNotFoundError(f'Scripps {code} file(s) not found: {message}')
        if daily:  # intermittent daily observations
            header = None  # header commented out
            names_std = ['date', 'time', 'excel', 'decimal']
            names_idx = names_idx if in_situ_mlo else names_std
            names_use = ['mean', 'count', 'flag']  # skip useless scale constant
            names_data = ['mean', 'count', 'scale'] if in_situ_mlo else ['count', 'flag', 'mean']   # noqa: E501
        else:
            header = (0, 1, 2)  # multi-index header
            names_idx = ['year', 'month', 'excel', 'decimal']
            names_use = ['mean', 'trend']  # used columns
            names_skip = [f'skip_{idx}' for idx in range(4)]  # ignored columns
            names_data = names_use + names_skip if raw else names_skip + names_use
            names_data += ['station'] if in_situ_mlo else []  # skip useless station code  # noqa: E501

    # Read and format data
    # NOTE: Pandas does not accept default 'names=None', and while frames can have
    # distinct column dtypes, casting to '.values' converts mixed types to 'object'.
    # NOTE: Cannot overwrite multi-index columns generated with e.g. header=[0, 1]
    # using e.g. names=[...] (raises error) and 'skiprows' is not comment-relative
    # for now. See: https://github.com/pandas-dev/pandas/issues/23067
    # data = data.interpolate(method='linear', limit_area='inside')  # use native filled
    # kwargs.update({'names': names, 'usecols': usecols} if names else {})
    names_sort = ['mean', 'mean sigma', 'count', 'flag']
    names_use = [name for name in names_use or names_data if name in names_data]
    idxcols = tuple(range(len(names_idx)))  # TODO: submit report for multi-index bug
    usecols = [name for name in names_sort if name in names_use]
    usecols += [name for name in names_use if name not in usecols]
    kwargs = dict(
        sep=sep,  # possibly includes whitespace
        comment=comment,
        header=header,
        index_col=idxcols,
        engine='python',  # support regex separator
    )
    data = pd.read_csv(paths[0], **kwargs)
    data.index.names = names_idx
    data.columns = pd.Index(names_data)
    for name in names_data:
        if not np.issubdtype(data[name].dtype, np.number):
            continue
        if name in ('count', 'flag'):  # fill erroneous NaNs and convert to integer
            data[name] = data[name].fillna(0).astype(int)
        else:  # see mauna loa file, may be 0.00 -0.99 or -9.99
            data[name] = data[name].mask(data[name] <= 0)
    if not esrl and not daily:
        locs = [names_data.index(name) for name in ('mean', 'trend')]
        for loc in range(2):  # restrict spline interpolation to observing period
            idxs, = np.where(~data.iloc[:, loc].isnull())  # avoid object caseting
            idx1, idx2 = min(idxs, default=0), max(idxs, default=data.index.size)
            data.iloc[:idx1, locs[loc]] = np.nan
            data.iloc[idx2 + 1:, locs[loc]] = np.nan
    if decimal:  # retain only decimal date
        levels = set(data.index.names) - {'decimal', 'decimal date'}
        for level in levels:
            data = data.droplevel(level, axis=0)
        data.index.name = 'year'
    else:  # convert to datetime index
        index_names = data.index.names
        times = [
            'T'.join(time[index_names.index(n)] for n in ('date', 'time'))
            if 'date' in index_names and 'time' in index_names
            else time[index_names.index('date')]
            if 'date' in index_names
            else f'{time:04d}-01-01'  # calculated for january
            if 'month' not in index_names
            else f'{time[0]:04d}-{time[1]:02d}-01'  # easier for 'MS' resampling later
            if 'day' not in index_names
            else f'{time[0]:04d}-{time[1]:02d}-{time[2]:02d}'
            for time in data.index.values
        ]
        index = np.array(times, dtype='M')
        index = pd.Index(index, name='time')
        data = data.set_index(index)
    data = data[usecols]  # remove unused columns
    data.name = STATION_NAMES[code]
    data.label = STATION_LABELS[code]
    units = 'ppm / year' if growth else 'ppm'
    data = ureg.Quantity(data, units)
    return data
