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


# Scripps station data
# NOTE: Have manually inspected stations and moved records with severe shortages or
# multi-year gaps in critical 2000 to ~2020 period to 'incomplete' folder.
# NOTE: Unlike GML data these are all marine sites and begin relatively early, and
# all are near Pacific or Southern Ocean. In future should support regional averaging.
# See: https://scrippsco2.ucsd.edu/data/atmospheric_co2/sampling_stations.html
# See: https://scrippsco2.ucsd.edu/data/atmospheric_co2/primary_mlo_co2_record.html
# 'fan': 'fanning',  # only covers 1970s through 1980s
# 'fan': ('Fanning Island', 'KI'),
# 'fan': (3.9, -159.4, 3, 1977),
# 'chr': 'christmas',  # missing data 2015 through 2016 (daily data implies even worse)
# 'chr': ('Christmas Island', 'KI'),
# 'chr': (2.0, -157.3, 2, 1974),
SCRIPPS_NAMES = {
    'gl': 'globe',  # starts 1979
    'alt': 'alert',
    'ptb': 'barrow',
    'ljo': 'la-jolla',
    'mlo': 'mauna-loa',
    'kum': 'kumukahi',
    'sam': 'samoa',
    'ker': 'kermadec',
    'nzd': 'baring',
    'spo': 'south-pole',
}
SCRIPPS_LABELS = {
    'gl': ('global average', None),
    'alt': ('Alert, Nunavut', 'CA'),  # see below for code translations
    'ptb': ('Barrow, Alaska', 'US'),
    'ljo': ('La Jolla Pier, California', 'US'),
    'mlo': ('Mauna Loa Observatory, Hawaii', 'US'),
    'kum': ('Cape Kumukahi, Hawaii', 'US'),
    'sam': ('American Samoa', 'AS'),
    'ker': ('Kermadec Island', 'NZ'),
    'nzd': ('Baring Head', 'NZ'),
    'spo': ('South Pole', 'AQ'),
}
SCRIPPS_COORDS = {  # (lat, lon, alt, start)
    'gl': (None, None, None, None),
    'alt': (82.3, -62.3, 210, 1985),
    'ptb': (71.3, -156.6, 11, 1961),
    'ljo': (32.9, -117.3, 10, 1957),
    'mlo': (19.5, -155.6, 3397, 1958),  # both esrl and scripps
    'kum': (19.5, -154.8, 3, 1979),
    'sam': (-14.2, -170.6, 30, 1981),
    'ker': (-29.2, -177.9, 2, 1982),
    'nzd': (-41.4, 174.9, 85, 1977),
    'spo': (-90.0, 0.0, 2810, 1957),
}
SCRIPPS_CODES = {name: code for code, name in SCRIPPS_NAMES.items()}

# GML station data
# See: https://gml.noaa.gov/dv/site/?program=ccgg
# See: https://gml.noaa.gov/aftp/data/trace_gases/co2/flask/surface/txt/
# NOTE: This is also included in file headers and repeated along columns. The webpage
# copy paste approach is consistent with scripps but should consider parsing metadata.
# NOTE: Only included stations that provide monthly average CO2 data (and first removed
# inactive stations by clicking 'show only active cites' on above page then manually
# checking and deleting files downloaded from ftp folders that are not listed.
# 'chr': 'christmas',  # missing data 2015 through 2016 (daily data implies even worse)
# 'chr': ('Christmas Island', 'KI'),  # Kiribati
# 'chr': (1.700, -157.152, 0.0, '-10 hours'),
ESRL_NAMES = {
    'gl': 'globe',  # starts 1979
    'alt': 'alert',
    'zep': 'ny-alesund',
    'sum': 'summit',
    'brw': 'barrow',
    'ice': 'storhofdi',
    'cba': 'cold-bay',
    'mhd': 'mace-head',
    'shm': 'shemya',
    'hun': 'hegyhatsal',
    'uum': 'ulaan',
    'nwr': 'niwot',
    'uta': 'wendover',
    'azr': 'terceira',
    'wlg': 'waliguan',
    'tap': 'tae-ahn',
    'bmw': 'tudor',
    'wis': 'weizmann',
    'mid': 'sand',
    'izo': 'izana',
    'key': 'biscayne',
    'ask': 'assekrem',
    'mlo': 'mauna-loa',
    'mko': 'mauna-kea',
    'kum': 'kumukahi',
    'rpb': 'ragged',
    'gmi': 'mariana',
    'sey': 'mahe',
    'asc': 'ascension',
    'smo': 'samoa',
    'nmb': 'gobabeb',
    'eic': 'easter',
    'cgo': 'cape-grim',
    'bhd': 'baring',
    'crz': 'crozet',
    'ush': 'ushuaia',
    'psa': 'palmer',
    'syo': 'syowa',
    'hba': 'halley',
    'spo': 'south-pole',
}
ESRL_LABELS = {
    'gl': ('global average', None),
    'alt': ('Alert, Nunavut', 'CA'),  # Canada
    'zep': ('Ny-Alesund, Svalbard', 'SJ'),  # Svalbard
    'sum': ('Summit Station', 'GL'),  # Greenland
    'brw': ('Barrow, Alaska', 'US'),  # United States
    'ice': ('Storhofdi, Vestmannaeyjar', 'IS'),  # Iceland
    'cba': ('Cold Bay, Alaska', 'US'),
    'mhd': ('Mace Head, County Galway', 'IE'),  # Ireland
    'shm': ('Shemya Island, Alaska', 'US'),
    'hun': ('Hegyhatsal', 'HU'),  # Hungary
    'uum': ('Ulaan Uul', 'MN'),  # Mongolia
    'nwr': ('Niwot Ridge, Colorado', 'US'),
    'uta': ('Wendover, Utah', 'US'),
    'azr': ('Terceira Island, Azores', 'PT'),  # Portugal
    'wlg': ('Mt. Waliguan', 'CN'),  # China
    'tap': ('Tae-ahn Peninsula', 'KR'),  # Korea
    'bmw': ('Tudor Hill, Bermuda', 'GB'),  # United Kingcom
    'wis': ('Weizmann Institute, Ketura', 'IL'),  # Israel
    'mid': ('Sand Island, Midway', 'UM'),  # US Minor Outlying Islands
    'izo': ('Izana, Tenerife, Canary Islands', 'ES'),  # Spain
    'key': ('Key Biscayne, Florida', 'US'),
    'ask': ('Assekrem', 'DZ'),  # Algeria
    'mlo': ('Mauna Loa, Hawaii', 'US'),
    'mko': ('Mauna Kea, Hawaii', 'US'),
    'kum': ('Cape Kumukahi, Hawaii', 'US'),
    'rpb': ('Ragged Point', 'BB'),  # Barbados
    'gmi': ('Mariana Islands', 'GU'),  # Guam
    'sey': ('Mahe Island', 'SC'),  # Seychelles
    'asc': ('Ascension Island', 'SH'),  # GB Ascension Island
    'smo': ('Tutuila', 'AS'),  # American Samoa
    'nmb': ('Gobabeb', 'NA'),  # Namibia
    'eic': ('Easter Island', 'CL'),  # Chile
    'cgo': ('Cape Grim, Tasmania', 'AU'),  # Australia
    'bhd': ('Baring Head', 'NZ'),  # New Zealand
    'crz': ('Crozet Island', 'FR'),  # France
    'ush': ('Ushuaia', 'AR'),  # Argentina
    'psa': ('Palmer Station', 'AQ'),  # US Antarctica
    'syo': ('Syowa Station', 'AQ'),  # JP Antarctica
    'hba': ('Halley Station', 'AQ'),  # GB Antarctica
    'spo': ('South Pole', 'AQ'),  # US Antarctica
}
ESRL_COORDS = {
    'gl': (None, None, None, None),
    'alt': (82.451, -62.507, 185.0, '-5 hours'),
    'zep': (78.907, 11.888, 474.0, '+1 hours'),
    'sum': (72.596, -38.422, 3209.5, '-2 hours'),
    'brw': (71.323, -156.611, 11.0, '-9 hours'),
    'ice': (63.400, -20.288, 118.0, '+0 hours'),
    'cba': (55.210, -162.720, 21.3, '-9 hours'),
    'mhd': (53.326, -9.899, 5.0, '+0 hours'),
    'shm': (52.711, 174.126, 23.0, '-10 hours'),
    'hun': (46.956, 16.652, 248.0, '+1 hours'),
    'uum': (44.452, 111.096, 1007.0, '+8 hours'),
    'nwr': (40.053, -105.586, 3523.0, '-7 hours'),
    'uta': (39.902, -113.718, 1327.0, '-7 hours'),
    'azr': (38.766, -27.375, 19.0, '-1 hours'),
    'wlg': (36.288, 100.896, 3810.0, '+8 hours'),
    'tap': (36.738, 126.133, 16.0, '+9 hours'),
    'bmw': (32.265, -64.879, 30.0, '-4 hours'),
    'wis': (29.965, 35.060, 151.0, '+2 hours'),
    'mid': (28.219, -177.368, 4.6, '-11 hours'),
    'izo': (28.309, -16.499, 2372.9, '+0 hours'),
    'key': (25.665, -80.158, 1.0, '-5 hours'),
    'ask': (23.262, 5.632, 2710.0, '+1 hours'),
    'mlo': (19.536, -155.576, 3397.0, '-10 hours'),
    'mko': (19.823, -155.469, 4199.0, '-10 hours'),
    'kum': (19.561, -154.888, 8.0, '-10 hours'),
    'rpb': (13.165, -59.432, 15.0, '-4 hours'),
    'gmi': (13.386, 144.656, 0.0, '+10 hours'),
    'sey': (-4.682, 55.532, 2.0, '+4 hours'),
    'asc': (-7.967, -14.400, 85.0, '-1 hours'),
    'smo': (-14.247, -170.564, 42.0, '-11 hours'),
    'nmb': (-23.580, 15.030, 456.0, '+1 hours'),
    'eic': (-27.160, -109.428, 47.0, '-7 hours'),
    'cgo': (-40.683, 144.690, 94.0, '+10 hours'),
    'bhd': (-41.408, 174.871, 85.0, '+12 hours'),
    'crz': (-46.434, 51.848, 197.0, '+5 hours'),
    'ush': (-54.848, -68.311, 12.0, '-3 hours'),
    'psa': (-64.774, -64.053, 10.0, '-3 hours'),
    'syo': (-69.013, 39.590, 14.0, '+3 hours'),
    'hba': (-75.605, -26.210, 30.0, '-2 hours'),
    'spo': (-89.980, -24.800, 2810.0, '+12 hours'),
}
ESRL_CODES = {name: code for code, name in ESRL_NAMES.items()}


def load_stations(
    *stations, annual=None, growth=None, daily=None, source=None, **kwargs,
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
    # TODO: Support global average from area-weighted station averages. Will require
    # some sophisticated algorithm that weights by isolation of the station.
    # TODO: Support loading in situ and flask / scripps and gml into same table. Will
    # require translating between scripps and esrl style codes. For now leave to later.
    annual = not daily if daily is not None else growth if annual is None else annual
    source = source or ('esrl' if annual else 'gml')
    names = SCRIPPS_NAMES if source == 'scripps' else ESRL_NAMES
    codes = SCRIPPS_CODES if source == 'scripps' else ESRL_CODES
    codes = [codes.get(station, station) for station in stations]
    codes = codes or (('gl', 'mlo') if source == 'esrl' else list(names))
    items = [code for code in names if code in codes]
    items += [code for code in codes if code not in names]
    datas, attrs, units = {}, {}, None
    kwargs.update(annual=annual, daily=daily, growth=growth)
    for code in items:
        isource = None if code == 'gl' else source
        data = load_station(code, source=isource, **kwargs)
        name = data.name  # station name
        if units is None:
            data, units = data.magnitude, data.units
        else:
            data = data.to(units).magnitude
        datas[name] = data
        attrs[name] = data.attrs
    data = pd.concat(datas.values(), keys=datas.keys(), axis='columns')
    data.attrs = attrs
    data = ureg.Quantity(data, units)  # WARNING: must come after attrs assignment
    return data


def load_station(
    station=None,
    base=None,
    annual=None,
    growth=False,
    daily=False,
    decimal=False,
    flasks=None,
    source=None,
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
    source : bool, optional
        Whether to load GML, ESRL, or Scripps observartions.
    raw : bool, optional
        Whether to load raw data instead of interpolated data. Not availble for ESRL.

    Returns
    -------
    data : pandas.DataFrame
        The station data.
    """
    # Station settings
    # TODO: Switch to pint-pandas support for quantifying underlying data
    # TODO: Read and translate commented-out headers instead of brute force method
    # NOTE: ESLR and Scripps have generally consistent format but some stations
    # still have exceptions. Should add to the below when reading new sources.
    station = station or 'mlo'
    annual = growth if annual is None else annual and not daily
    source = source or ('esrl' if annual or station in ('gl', 'globe') else 'gml')
    codes = SCRIPPS_CODES if source == 'scripps' else ESRL_CODES
    names = SCRIPPS_NAMES if source == 'scripps' else ESRL_NAMES
    labels = SCRIPPS_LABELS if source == 'scripps' else ESRL_LABELS
    coords = SCRIPPS_COORDS if source == 'scripps' else ESRL_COORDS
    base = Path(base or f'~/data/{source}-co2').expanduser()
    names_idx = ['year', 'month', 'day']
    index_cols = names_data = names_use = None
    if (source := source.lower()) not in ('gml', 'esrl', 'scripps'):
        raise ValueError(f'Invalid station source {source!r}.')
    if (code := codes.get(station, station)) not in names:
        raise ValueError(f'Invalid station name or code {station!r}.')
    if source == 'esrl':  # esrl data
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
            names_ann = ['trend', 'trend sigma'] if growth else ['mean', 'mean sigma']
            names_idx = ['year'] if annual else ['year', 'month', 'decimal']
            names_data = names_ann if annual else names_gl if code == 'gl' else names_ml
    elif source == 'gml':  # individual stations
        sep = r'\s+'  # standard table
        freq = 'day' if daily else 'month'
        modes = ('insitu', 'flask')  # carbon data
        modes = modes if flasks is None else ('flask',) if flasks else ('insitu',)
        freqs = {('month', 'insitu'): 'MonthlyData', ('day', 'insitu'): 'DailyData'}
        freqs = {**freqs, ('month', 'flask'): 'month', ('day', 'flask'): 'event'}
        files = [f'co2_{code}_surface-{mode}_1_ccgg_{freqs[freq, mode]}.txt' for mode in modes]  # noqa: E501
        paths = [base / file for file in files if (base / file).is_file()]
        flasks = paths and 'flask' in paths[0].name  # used below
        comment = '#'  # comment character
        if growth or annual:
            raise ValueError('Scripps does not provide annual data or growth rates.')
        if not paths:
            message = ', '.join(f'{base / file!s}' for file in files)
            raise FileNotFoundError(f'GML {code} file(s) not found: {message}')
        if not daily and flasks:
            header = None
            names_idx = ['year', 'month']
            index_cols = (1, 2)
            names_data = ['station', 'mean']
            names_use = ['mean']
        else:  # general format
            header = 0  # header row
            names_idx += ['hour', 'minute', 'second', 'datetime', 'decimal', 'other']
            names_skip = [f'skip_{idx}' for idx in range(9 if flasks else 5)]
            index_cols = range(1, len(names_idx) + 1)
            names_data = ['station', 'mean', 'mean sigma']
            names_data += ([] if flasks else ['count']) + names_skip + ['flag']
            names_use = ['mean', 'mean sigma', 'count']
            names_use += ['flag'] if flasks else []  # in situ flags provide no info
    else:  # scripps data
        in_situ_mlo = not flasks and code == 'mlo'  # WARNING: strange special case
        sep = r',\s*'  # WARNING: mlo files require appending ',' to header columns
        freq = 'daily' if daily else 'monthly'
        modes = ('merge', 'in_situ', 'flask')
        modes = modes if flasks is None else ('flask',) if flasks else ('in_situ',)
        files = [f'{freq}_{method}_co2_{code}.csv' for method in modes]
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
    idxcols = tuple(index_cols or range(len(names_idx)))
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
        is_number = np.issubdtype(data[name].dtype, np.number)
        if not is_number and name != 'flag':  # ignore non-numeric columns
            continue
        if not is_number:  # combine rejection/selection columns and remove info column
            data[name] = data[name].str.replace(r'(\.|.\Z)', '', regex=True)
        elif name in ('count', 'flag'):  # fill erroneous NaNs and convert to integer
            data[name] = data[name].fillna(0).astype(int)
        else:  # see mauna loa file, may be 0.00 -0.99 or -9.99
            data[name] = data[name].mask(data[name] <= 0)
    if not daily and source == 'scripps':
        locs = [names_data.index(name) for name in ('mean', 'trend')]
        for idx in range(2):  # restrict spline interpolation to observing period
            raw = data.iloc[:, idx]  # raw data always on first two columns
            idxs, = np.where(~raw.isnull())  # avoid casting object arrays
            idx1, idx2 = min(idxs, default=0), max(idxs, default=data.index.size)
            data.iloc[:idx1, locs[idx]] = np.nan
            data.iloc[idx2 + 1:, locs[idx]] = np.nan
    if decimal:  # retain only decimal date
        levels = set(data.index.names) - {'decimal', 'decimal date'}
        for level in levels:
            data = data.droplevel(level, axis=0)
        data.index.name = 'year'
    else:  # convert to datetime index
        index_names = data.index.names
        center = 1 if growth else 7
        times = [
            time[index_names.index('datetime')] if 'datetime' in index_names
            else 'T'.join(time[index_names.index(n)] for n in ('date', 'time'))
            if 'date' in index_names and 'time' in index_names
            else time[index_names.index('date')]
            if 'date' in index_names
            else f'{time:04d}-{center:02d}-01'  # central month for data
            if 'month' not in index_names
            else f'{time[0]:04d}-{time[1]:02d}-01'  # easier for 'MS' resampling later
            if 'day' not in index_names
            else f'{time[0]:04d}-{time[1]:02d}-{time[2]:02d}'
            for time in data.index.values
        ]
        index = np.array(times, dtype='M')
        index = pd.Index(index, name='time')
        data = data.set_index(index)
    name, attrs = names[code], {}  # xarray settings
    label, region = labels[code]
    lon, lat, alt, _ = coords[code]
    units = 'ppm / year' if growth else 'ppm'
    attrs['lat'] = ureg.degN * (lon or 0)
    attrs['lon'] = ureg.degE * (lat or 0)
    attrs['alt'] = ureg.m * (alt or 0)
    attrs['code'] = code.upper()  # station code
    attrs['method'] = 'flask' if 'flask' in paths[0].name else 'in situ'
    attrs['source'] = source.upper()  # data source
    attrs['long_name'] = label + (region and f' ({region})' or '')
    attrs['short_name'] = label.split(',')[0]  # exclude subregion
    data = data[usecols]  # remove unused columns
    data.name = name
    data.attrs = attrs
    data = ureg.Quantity(data, units)  # WARNING: must come after attrs assignment
    return data
