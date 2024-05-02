#!/usr/bin/env python3
"""
Processing utilities for pandas tables.
"""
# TODO: Merge with 'arrays.py' and support arbitrary input types. Should leverage
# public facing abstract-array climopy wrappers.
from datetime import datetime

import numpy as np
import pandas as pd
from climopy import const, ureg, vreg  # noqa: F401
from climopy import var
from icecream import ic  # noqa: F401

__all__ = [
    'to_datetime',
    'to_monthly',
    'get_growth',
    'restrict_time',
    'detrend_time',
    'reduce_time',
    'select_time',
]

# Season months
# Critical to double so e.g. 'DJF' is allowed
SEASON_MONTHS = 2 * 'JFMAMJJASOND'

# Convert ppm to mass by multiplying molar mass of CO2 by moles of entire atmosphere
# (mass divided by dry air molar mass). See: Ballantyne 2012 Methods (differs by 0.1%)
MOLAR_TO_MASS = const.Mc * (const.Ma / const.Md)  # (kg / mol) * kg / (kg / mol)


def to_datetime(data):
    """
    Translate annual averages or growth rates to datetime indices.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data with year index.

    Returns
    -------
    result : pandas.DataFrame
        The output data with datetime index.
    """
    # NOTE: The budget terms e.g. ESRL CO2 actually represent changes between december
    # january average of each year end so should be centered on date 01-01 not 07-01.
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f'Invalid input type {type(data)}.')
    if hasattr(data.index, 'year'):
        index = data.index.copy()
    else:
        index = data.index.map(lambda year: np.datetime64(f'{year:04d}-01-01'))
    result = data.copy()
    result.index = index
    result.index.name = 'time'
    result = result if units is None else ureg.Quantity(result, units)
    return result


def to_monthly(data):
    """
    Translate annual averages or growth rates to monthly resolution.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data with year index.

    Returns
    -------
    result : pandas.DataFrame
        The output data with monthly frequency.
    """
    # NOTE: Annual growth rate indices indicate starting date for next 12 months (e.g.
    # growth from January 1st to December 31st has index January 1st). Similarly index
    # on monthly data should indicate growth over corresponding month. But center for
    # e.g. January 1st annual growth is July 1st, while center for e.g. July 1st
    # monthly growth is July 15th. Account for this by interpolating on month starts
    # (resample interpolate fails when dates are not on first) then average current
    # months with following months to 'interpolate' onto central day of each month.
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f'Invalid input type {type(data)}.')
    index = data.index
    data = to_datetime(data)
    func = lambda days: np.timedelta64(int(days), 'D')
    delta = index[1:] - index[:-1]
    monthly = np.all((delta >= func(28)) & (delta <= func(31)))
    annual = np.all((delta >= func(365)) & (delta <= func(366)))
    if annual:
        func = lambda time: np.datetime64(f'{time.year:04d}-07-01')
        data.index = index.map(func)
    if monthly:
        func = lambda time: np.datetime64(f'{time.year:04d}-{time.month:02d}-01')
        data.index = index.map(func)
    else:  # e.g. annual data or monthly data with missing slots
        data = data.resample('MS').interpolate('linear')
        index = data.index
        days = index.days_in_month.values
        days = np.expand_dims(days, -1) if data.ndim == 2 else days
        days0, data0 = days[:-1], data.iloc[:-1]
        days1, data1 = days[1:], data.iloc[1:].set_index(index[:-1])
        data = (days0 * data0 + days1 * data1) / (days0 + days1)
    result = data if units is None else ureg.Quantity(data, units)
    return result


def get_growth(data, delta=False):
    """
    Translate annual or monthly concentration to growth rates.

    Parameters
    ----------
    data : pandas.DataFrame
        The input concentrations.
    delta : bool, optional
        Whether to return growth differences relative to preceding year.

    Returns
    -------
    result : pandas.DataFrame
        The calculated growth rates.
    trend : pandas.DataFrame
        The best fit to the growth rates.
    """
    # NOTE: ESRL 'gr' data shows growth over the year shown on index (i.e. DJ average
    # minus DJ average), and this similarly gives growth rates for centers of months
    # shown on index (i.e. J is JF minus DJ), so that January to December average
    # of below growth rates reconstructs ESRL (CO2 from internal months cancel out).
    units = getattr(data, 'units', ureg.ppm)
    data = getattr(data, 'magnitude', data)
    data = to_monthly(restrict_time(data))
    if units.is_compatible_with('GtC / yr'):
        return ureg.Quantity(data, units).to('GtC / yr')
    if not units.is_compatible_with('ppm'):
        raise ValueError(f'Invalid input units {units!r}.')
    days = data.index.days_in_month.values
    days0, data0 = days[:-2], data.iloc[:-2].set_index(data.index[1:-1])
    days1, data1 = days[1:-1], data.iloc[1:-1]
    days2, data2 = days[2:], data.iloc[2:].set_index(data.index[1:-1])
    results = {}
    for column in data.columns:
        series0 = data0[column]
        series1 = data1[column]
        series2 = data2[column]
        if 'sigma' in column:
            sigma1 = (days1 * series1 ** 2 + days2 * series2 ** 2) / (days1 + days2)
            sigma0 = (days0 * series0 ** 2 + days1 * series1 ** 2) / (days0 + days1)
            result = np.sqrt(sigma1 + sigma0) / days1
        else:
            growth1 = (days1 * series1 + days2 * series2) / (days1 + days2)
            growth0 = (days0 * series0 + days1 * series1) / (days0 + days1)
            result = (growth1 - growth0) / days1
        if delta:  # change in growth over following year
            result = result - result.shift(12)
        results[column] = result
    result = pd.DataFrame(results)
    result = ureg.Quantity(result, units)
    result = result * MOLAR_TO_MASS / ureg.days  # WARNING: must come after Quantity
    result = result.to('GtC / yr')
    return result


def detrend_time(data, base=None):
    """
    Remove trend from the input time series.

    Parameters
    ----------
    data : pandas.DataFrame
        The input growth rates.
    base : pandas.DataFrame, optional
        The baseline concentration or emissions to subtract from the series.

    Parameters
    ----------
    result : pandas.DataFrame
        The detrended data.
    trend : pandas.DataFrame
        The linear trend(s).
    """
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    data = restrict_time(data)  # fill nulls before interpolation
    if base is None:
        days = data.index.days_in_month.values
        days = np.append(0, np.cumsum(days))
        days = 0.5 * (days[1:] + days[:-1])  # central month
        *_, trend, _, _ = var.linefit(days, data.values, axis=0, correct=False)
        if isinstance(data, pd.Series):
            trend = pd.Series(trend, index=data.index, name=data.name)
        elif isinstance(data, pd.DataFrame):
            trend = pd.DataFrame(trend, index=data.index, columns=data.columns)
        else:
            raise ValueError(f'Unexpected data type {data!r}.')
    else:
        if base.ndim == 1:
            base = base.to_frame()  # pint compatible
        if base.shape[1] == data.shape[1]:  # e.g. global CO2
            base.magnitude.columns = data.columns
        elif base.shape[1] == 1:
            base = pd.DataFrame({name: base for name in data.columns})
        else:  # e.g. forgot to select base column
            raise ValueError(f'Incompatible shapes {base.shape} and {data.shape}.')
        base = ureg.Quantity(base, units)  # translate units if quantity
        base = restrict_time(base.magnitude)
        locs = data.index.intersection(base.index)
        data, base = data.loc[locs], base.loc[locs]
        trend = base * data.values.mean().item() / base.values.mean().item()
    result = data - trend if units is None else ureg.Quantity(data - trend, units)
    trend = trend if units is None else ureg.Quantity(trend, units)
    return result, trend


def reduce_time(data, time=None, lag=None, trunc=None):
    """
    Reduce over time index using monthly selection or seasonal or annual averages.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The input data.
    time : int or str or sequence, optional
        The month or season or time frequency indicator.
    lag : int, optional
        The number of months to lag the data.
    trunc : int, optional
        The number of months to optionally truncate. If ``True`` truncate to one point.

    Returns
    -------
    result : array-like
        The reduced data.
    label : str
        A suitable coordinate label.
    name : str
        A suitable filename string.
    """
    # Parse input argument
    # TODO: Merge this with 'coupled/reduce' _get_datetime(), 'cmip_data.utils'
    # average_periods(), and climopy 'climopy/accessor' sel_time(). This has superset
    # of features and supports specifying generalized times with single argument.
    months = None  # initial months
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    time = time if time is not None else 'mon'
    data = to_monthly(data)
    data = data if lag is None else data.shift(-lag, freq='MS')
    if not isinstance(time, str):
        months = np.atleast_1d(time)
    elif time in ('ann', 'annual'):
        months = np.arange(1, 13)  # average starting january
    elif time in ('mon', 'monthly'):
        label, name = 'monthly', 'mon'
        data = data if units is None else ureg.Quantity(data, units)
        return data, label, name
    else:
        try:
            date = datetime.strptime(time[:3], '%b')
            months = np.array([date.month])
        except ValueError:
            pass
        try:
            idx = SEASON_MONTHS.index(time.upper())
            months = np.arange(idx + 1, idx + len(time) + 1)
        except ValueError:
            pass
    # Get annual or season average
    # NOTE: Here use '.mul' and '.div' since 'df * [array|index|series]' evaluates
    # along columns (i.e. row major order) and 'df1 * df2' matches column names.
    imax = len(months) - 1
    months = months % 12 + 1  # WARNING: must come after above season string
    if months is None:
        raise ValueError(f'Unknown time identifier {time!r}.')
    if months.size == 12:  # annual average
        label, name = 'annual', 'ann'
    elif months.size > 1:  # season average
        label = name = SEASON_MONTHS[months.min() - 1:months.max()]
    elif date := datetime(1800, months.item(), 1):  # single month
        label, name = date.strftime('%B'), date.strftime('%b').lower()
    if trunc is True:  # e.g. DJF --> J, NDJF --> DJ
        months = months[imax // 2:imax // 2 + 1 + (imax % 2)]
    else:  # truncate by input
        months = months[trunc:trunc and -trunc or None]
    loc0 = [t for t, time in enumerate(data.index) if time.month == months[0]]
    loc1 = [t for t, time in enumerate(data.index) if time.month == months[-1]]
    data = data.iloc[loc0[0]:loc1[-1] + 1]  # omit incomplete seasons
    data = data.iloc[[time.month in months for time in data.index]]
    days = data.index.days_in_month.to_series(name='days', index=data.index)
    func = lambda data: data.sum(axis=0, skipna=False)
    group = np.arange(data.shape[0]) // len(months)
    result = data.mul(days, axis='rows').groupby(group).apply(func)
    result = result.div(days.groupby(group).sum(), axis='rows')
    result.index = data.index[len(months) // 2::len(months)]  # centered months
    result = result if units is None else ureg.Quantity(result, units)
    return result, label, name


def restrict_time(data):
    """
    Restrict data to within non-null years then interpolate remaining nulls.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data.

    Returns
    -------
    result : pandas.DataFrame
        The restricted output data.
    """
    # NOTE: This is critical for e.g. climopy 'linefit' methods that rely on
    # no missing values. In future should add masked array features to climopy.
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(f'Invalid input type {type(data)}.')
    mask = data.isnull()
    mask = mask if mask.ndim == 1 else mask.any(axis=1)
    idxs, = np.where(~mask)
    idx1 = min(idxs, default=0)
    idx2 = max(idxs, default=data.index.size - 1)
    result = data.iloc[idx1:idx2 + 1]
    result = result.interpolate(method='linear', axis=0)
    result = result if units is None else ureg.Quantity(result, units)
    return result


def select_time(data, years=None):
    """
    Select input years or date range.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data.
    years : int or 2-tuple
        The number of trailing years or inclusive year bounds.

    Returns
    -------
    result : pandas.DataFrame
        The restricted output data.
    """
    # NOTE: This should come after reduce_time(). Will use january for the datetime
    # index of 'winter' averages e.g. 'DJF' or 'NDJF' so no special handling needed.
    index = data.index
    units = getattr(data, 'units', None)
    data = getattr(data, 'magnitude', data)
    data = restrict_time(data)
    year1 = getattr(index, 'year', index)[-1].item()
    years = (2000, None) if years is None else years
    years = np.atleast_1d(years)
    if years.size == 1:  # use e.g. (None, year) or (year, None) for specific ranges
        years = np.array([year1 - years[0], year1])
    if years.size != 2:
        raise ValueError(f'Unexpected growth sample years {years}.')
    if index.name == 'year':
        loc0, loc1 = years[0], years[1]
    else:  # datetime indexing
        loc0 = years[0] and f'{years[0]:04d}-01-01' or None
        loc1 = years[1] and f'{years[1]:04d}-12-31' or None
    result = data.loc[loc0:loc1]
    result = result if units is None else ureg.Quantity(result, units)
    return result
