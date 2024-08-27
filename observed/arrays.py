#!/usr/bin/env python3
"""
Processing utilities for xarray data arrays.
"""
# TODO: Merge with 'tables.py' and support arbitrary input types. Should leverage
# public facing abstract-array climopy wrappers. Note that all functions assume
# dequantified xarrays with units attributes instead of quantified arrays.
import math
import warnings
from datetime import datetime
from icecream import ic  # noqa: F401

import cftime
import numpy as np
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401


__all__ = [
    'annual_average',
    'annual_filter',
    'detrend_dims',
    'regress_dims',
]

# Message cache
# NOTE: Used to prevent duplicate trend messages
VERBOSE_MESSAGES = set()


def _get_mask(data, ocean=True):
    """
    Return ocean or land mask using `global_land_mask`.

    Parameters
    ----------
    data : xarray.DataArray
        The input array.
    ocean : bool, optional
        Whether to keep ocean or land.

    Returns
    -------
    mask : xarray.DataArray
        A boolean masking array.
    """
    from global_land_mask import globe
    lat = data.lat.values  # {{{
    lon = data.lon.values % 180  # convention
    lon, lat = np.meshgrid(lon, lat)  # creates lat x lon arrays
    if ocean:  # ocean only
        mask = globe.is_ocean(lat, lon)
    else:  # land only
        mask = globe.is_land(lat, lon)
    coords = {'lat': data.lat, 'lon': data.lon}
    mask = xr.DataArray(mask, dims=('lat', 'lon'), coords=coords)
    return mask


def to_decimal(data, dim=None, coords=None):
    """
    Convert datetime index to decimal year.

    Parameters
    ----------
    data : xarray.DataArray
        The input data with datetime coordinate.
    dim : str, default: 'time'
        The datetime coordinate name.
    coords : bool, optional
        Whether to change data or coordinates. Default is both.

    Returns
    -------
    result : xarray.DataArray
        The array with decimal year coordinate.
    """
    # TODO: Consider moving this to 'climopy' or 'coupled' (more relevant for models).
    # NOTE: Xarray cftime has 'dayofyear' 'days_in_month' and 'is_leap_year' but no
    # 'days_in_month' so derive from calendar: https://unidata.github.io/cftime/api.html
    dim = dim or 'time'  # {{{
    if dim not in data.dims:
        raise ValueError(f'Time dimension {dim!r} is not in {data.dims}.')
    index = data.indexes[dim]
    if not isinstance(index, (xr.CFTimeIndex, pd.DatetimeIndex)):
        return data  # assume already decimal year
    date = data.coords[dim]
    calendar = date.dt.calendar
    if isinstance(index, pd.DatetimeIndex):
        days = 365 + date.dt.is_leap_year.astype(int)
    elif calendar in ('360_day',):
        days = xr.full_like(date, 360, dtype=int)
    elif calendar in ('365_day', 'noleap'):
        days = xr.full_like(date, 365, dtype=int)
    elif calendar in ('366_day', 'all_leap'):
        days = xr.full_like(date, 366, dtype=int)
    else:  # no .dt.is_leap_year method
        days = [cftime.is_leap_year(t, calendar) for t in date.dt.year.values]
        days = np.array(days, dtype=int)
        days = 365 + xr.DataArray(days, dims=dim)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        year = date.dt.year + date.dt.dayofyear / days
    year.name = date.name
    year.attrs.update(date.attrs)
    year.attrs['units'] = 'year'
    if not coords and data.name == date.name:
        result = year  # change data not coordinates
    elif coords or coords is None:
        result = data  # change coordinates
    else:
        raise ValueError(f'Invalid data {data.name!r} for option {coords=}.')
    if coords or coords is None:
        result = result.assign_coords({dim: year})
    return result


def annual_average(data, **kwargs):
    """
    Return annual averages computed starting from the input month.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.
    **kwargs
        Passed to `annual_filter`.

    Returns
    -------
    result : xarray.DataArray
        The averaged data.
    """
    # WARNING: Critical to drop 'time_bnds' bounds or else trigger error trying to get
    # weighted averages of datetimes. Solved with open_dataarray(decode_coords='all')
    # NOTE: See cmip_data.utils.average_periods for the groupby() method (also in
    # coupled.process.apply_reduce and cmip_data.feedbacks._feedbacks_from_fluxes).
    # Here instead use resample for march-centered averages and to preserve times.
    # See: https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    # See: https://docs.xarray.dev/en/stable/generated/xarray.cftime_range.html
    name = data.name  # {{{
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        months = data.time.dt.month.values
    if len(np.unique(months)) == 1:
        return data
    data = annual_filter(data, **kwargs)
    month = data.time[0].dt.strftime('%b').item().upper()
    with warnings.catch_warnings():  # datetime64 emits annoying warning
        warnings.simplefilter('ignore')
        days = data.time.dt.days_in_month  # do not normalize yet, weight for resample()
        data = data.astype(data.dtype)  # preserve data type
        with xr.set_options(keep_attrs=True):
            numer = (data * days).resample(time=f'AS-{month}')
            denom = (xr.ones_like(data) * days).resample(time=f'AS-{month}')
            result = numer.sum(dim='time', skipna=False) / denom.sum(dim='time')
    result.name = name
    result.time.attrs.update(data.time.attrs)  # critical for add_cell_measures()
    return result


def annual_filter(data, years=None, month=None, anomaly=False):
    """
    Return data filtered to include only full years.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.
    years : tuple or slice, optional
        The optional years to use for filtering.
    month : int or str, optional
        The starting month for the averages.
    anomaly : bool, optional
        Whether to return monthly anomalies.

    Returns
    -------
    result : xarray.DataArray
        The filtered data.
    """
    # NOTE: Critical even for monthly regressions to prevent e.g. residual seasonal
    # component in either the climate or the forced response from influencing results.
    with warnings.catch_warnings():  # {{{
        warnings.simplefilter('ignore')
        months = data.time.dt.month.values
    if years is None:
        years = (None, None)
    elif isinstance(years, slice):
        years = (years.start, years.stop)
    elif np.isscalar(years) or len(years) != 2:
        raise TypeError(f'Invalid years specification {years!r}.')
    if (count := len(np.unique(months))) == 1:
        return data.isel(time=slice(*years))
    elif count != 12:
        raise TypeError(f'Invalid number of unique months {count}.')
    if month is None:
        month = data.time[0].dt.month.item()  # integer month
    elif isinstance(month, str):
        month = datetime.strptime(month.lower()[:3], '%b').month
    else:
        month = int(month)  # integer month
    idxs1, = np.nonzero(months == month)
    idxs2, = np.nonzero(months == (month - 2) % 12 + 1)  # 1 --> 12
    years = (years[0] and years[0] * 12, years[1] and years[1] * 12)
    result = data.isel(time=slice(idxs1[0], idxs2[-1] + 1))
    result = data.isel(time=slice(*years))
    if anomaly:  # note data already in anomaly form, but this uses selected period
        climate = result.groupby('time.month').mean()
        with xr.set_options(keep_attrs=True):
            result = result.groupby('time.month') - climate
        result = result.drop_vars('month')
    return result


def detrend_dims(data, dim=None, verbose=False, **kwargs):
    """
    Return detrended input data.

    Parameters
    ----------
    data : xarray.DataArray
        The input data.
    dim : str, default: 'time'
        The detrending dimension.
    verbose : bool, optional
        Whether to print detrend information. Default is ``False``.
    **kwargs
        Passed to `regress_dims`.

    Returns
    -------
    result : xarray.DataArray
        The detrended data.
    """
    # WARNING: Critical to auto-drop regress_dims 'x', 'y', 'fit', 'fit1', 'fit2'
    # NOTE: For regional data this will remove regional trends. Then when taking
    # global average will reproduce removal of global average trend.
    dim = dim or 'time'  # {{{
    coord = getattr(data, dim or 'time')  # possibly integer
    idata = kwargs.pop('coords', None)
    kwargs['nofit'] = False
    kwargs['coords'] = None if idata is None else getattr(idata, dim)
    result = regress_dims(coord, data, dim=dim, **kwargs)
    if kwargs.get('nobnds', None):
        slope, *fits, rsq, dof = result
    else:  # regression fit bounds
        slope, _, _, *fits, rsq, dof = result
    name = data.name
    attrs = data.attrs.copy()
    slope = slope.squeeze(drop=True)
    if verbose:  # print message
        if 'lon' in dof.sizes and 'lat' in dof.sizes:
            dof = dof.climo.add_cell_measures()
            dof = dof.climo.average('area')  # 'units' can now be missing
        if 'lon' in slope.sizes and 'lat' in slope.sizes:
            slope = slope.climo.add_cell_measures()
            slope = slope.climo.average('area')
        head = name or 'unknown'
        beta = f'{10 * slope.item(): >5.2f}'
        unit = attrs.get('units', None)
        msg = f'{head: >8s} trend: {beta} {unit} / decade'
        msg = f'{msg} (dof {dof.item():.0f})'
        if msg not in VERBOSE_MESSAGES:
            print(msg)
        VERBOSE_MESSAGES.add(msg)
    drop = ('xdata', 'ydata', 'ymean')
    keys = ('fit', 'fit1', 'fit2')
    mean = fits[0].ymean.drop_vars('ymean')
    fits = [fit.drop_vars(drop, errors='ignore') for fit in fits]
    coord = {key: fit for key, fit in zip(keys, fits)}  # possibly only 'fit'
    data = data if idata is None else idata
    data = mean + data - fits[0]  # WARNING: must come after drop
    data.name = name
    data.attrs.update(attrs)
    data.coords.update(coord)
    return data


def regress_dims(
    denom, numer, dim=None, raw=False, stat=None, ocean=None, weights=None, coords=None,
    correct=None, pctile=True, manual=False, nobnds=False, nofit=None, nofitbnds=None,
):
    """
    Return regression parameters.

    Parameters
    ----------
    denom, numer : xarray.DataArray
        The regression denominator and numerator.
    dim : str, default: 'time'
        The regression dimension(s). Parsed by climopy.
    raw : bool, optional
        Whether to skip removing the mean before the projection.
    stat : {'slope', 'proj', 'covar', 'corr', 'rsq'}, optional
        Whether to return the input statistic instead of slope.
    ocean : bool, optional
        Whether to mask to only ocean or only land.
    weights : array-like, optional
        The manual weights used for the fit. Default is nothing.
    coords : array-like, optional
        The coordinates used for the fit. Default is `denom`.

    Other Parameters
    ----------------
    correct : bool or str, optional
        Whether to correct the standard error for reduced degrees of freedom due
        to serial correlation. If sequence then ``correct`` coordinate is added.
    pctile : bool or float, optional
        Whether to return percentiles bounds or standard error deviations. If float
        this is the percentile range. If sequence then ``pctile`` coordinate is added.
    manual : bool, optional
        Whether to skip automatic weighting by cell measures (e.g. duration) and infer
        degrees of freedom from input weights (if provided) instead of sample count.
    nofit : bool, optional
        Whether to skip calculating regression fit. If ``True`` the three fit arguments
        are not returned. Default is ``True`` if the statistic is not `'slope'`.
    nobnds : bool, optional
        Whether to skip calculating statistic bounds. If ``True`` the four lower and
        upper bound arguments are not returned.
    nofitbnds : bool, optional
        Whether to skip calculating fit bounds. If ``True`` the two fit bound arguments
        are not returned and the denominator coordinate is not sorted.

    Returns
    -------
    data : xarray.DataArray
        The calculated statistic.
    data1, data2 : xarray.dataArray, optional
        The statistic uncertainty bounds. Skipped if `nobnds` is ``True``.
    fit : xarray.DataArray, optional
        The regression relationship fitting. Skipped if `nofit` is ``True``.
    fit1, fit2 : xarray.DataArray, optional
        The uncertainty bounds. Skipped if `nobnds`, `nofit`, or `nofitbnds`.
    rsq, dof : xarray.DataArray, optional
        The degrees of freedom and variance explained. Skipped if `nobnds` and `nofit`.
    """
    # Get regression estimates
    # NOTE: Critical to auto-drop 'x', 'y', 'fit', 'fit_lower', and 'fit_upper' coords
    # left over from detrending or else get weird hard-to-debug errors (see below).
    # TODO: Also use this for e.g. weighted standard deviations, variance, and other
    # statistics. Should add to climopy and incorporate with covariance driver.
    # TODO: Move this into climopy linefit(). Currently 'institute-factor' does not
    # support percentile bounds, optionally disabling autocorrelation adjustment, or
    # optionally using other series for autocorrelation adjustment, and 'monde-updates'
    # does not support weighted regressions or weighted uncertainty bounds and does
    # not auto-detrend the series used for adjustment.
    denom, numer = denom.climo.dequantify(), numer.climo.dequantify()  # {{{
    dims, kw_parse = dim or 'time', {'include_scalar': False, 'include_pseudo': False}
    dims = denom.climo._parse_dims(dims, **kw_parse)  # must be in denominator
    dim = dims[0] if len(dims) == 1 else None
    nofit = stat and stat != 'slope' if nofit is None else nofit
    nosort = dim is None or denom.ndim > 1 or nofit or nobnds or nofitbnds
    if not nofit and stat and stat != 'slope':
        raise TypeError(f'Incompatible arguments {stat=} {nofit=}. Remove nofit.')
    if istime := dim == 'time' and denom.name == 'time':  # convert data array
        denom = to_decimal(denom, coords=False)
    if manual:  # input weights only not infer weights
        wgts = xr.ones_like(denom) if weights is None else weights
    else:  # infer automatically (_parse_weights raises error if missing)
        numer = numer.climo.add_cell_measures()
        dims, *parts = numer.climo._parse_weights(dims, weight=weights, **kw_parse)
        wgts = math.prod(wgt.climo.dequantify() for wgts in parts for wgt in wgts)
    if ocean is not None:  # TODO: improve this
        mask1 = _get_mask(denom, ocean=ocean)
        mask2 = _get_mask(numer, ocean=ocean)
        wgts = xr.where(mask1 | mask2, wgts, 0)
    denom = denom.drop_vars(denom.coords.keys() - denom.sizes.keys())
    numer = numer.drop_vars(numer.coords.keys() - numer.sizes.keys())
    wgts = wgts.drop_vars(wgts.coords.keys() - wgts.sizes.keys())
    denom, numer, wgts = xr.align(denom, numer, wgts)  # align 'dims' coordinates
    denom, numer, wgts = xr.broadcast(denom, numer, wgts)  # broadcast other coordinates
    nulls = numer.isnull() | denom.isnull()
    numer, denom, wgts = numer.where(~nulls), denom.where(~nulls), wgts.where(~nulls)

    # Calculate regression statistic
    # NOTE: This defines projection 'standard error' as regression error formula times
    # sx and covariance 'standard error' as correlation error formula times sx * sy.
    # NOTE: For simplicity this gives the *biased* weighted variance estimator (analogue
    # for unweighted data is SS / N) instead of the complex *unbiased* weighted variance
    # estimator (analogue for unweighted data SS / N - 1). See full formula here:
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    # NOTE: Calculation of residual leverages facts that offset = ymean - slope * xmean
    # and slope stderr = sqrt(resid ** 2 / xanom ** 2 / n - 2) where resid =
    # y - ((ymean - slope * xmean) + slope * x) = (y - ymean) - slope * (x - xmean)
    # https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
    if manual:  # degrees of freedom from weights  # {{{
        size = wgts.sum(dims)
    else:  # normalize and get size
        size = math.prod(numer.sizes[key] for key in dims)
    wgts = wgts / wgts.sum(dims)  # now (w * d).sum() is effectively d.sum() / n
    davg = 0 if raw else (wgts * denom).sum(dims, skipna=True)
    navg = 0 if raw else (wgts * numer).sum(dims, skipna=True)
    danom, nanom = denom - davg, numer - navg
    covar = (wgts * danom * nanom).sum(dims, skipna=True)
    dvar = (wgts * danom ** 2).sum(dims, skipna=True)
    dstd = np.sqrt(dvar)
    slope = covar / dvar
    offset = navg - slope * davg
    if not nofit or not nobnds or stat == 'corr':
        nanom = wgts * (numer - navg) ** 2
        nvar = nanom.sum(dims, skipna=True)
        nstd = np.sqrt(nvar)
    if not stat or stat == 'slope':
        data = slope
        units = f'{numer.units} / {denom.units}'
    elif stat == 'cov' or stat == 'covar':
        data = covar
        units = f'{numer.units} {denom.units}'
    elif stat == 'proj':
        data = covar / dstd
        units = numer.units
    elif stat == 'corr':
        data = covar / (dstd * nstd)
        units = ''
    else:
        raise ValueError(f'Unknown input statistic {stat!r}.')
    units = units if denom.units else units and numer.units or ''
    data.attrs.update(units=units)
    if nobnds and nofit:
        return data

    # Get degrees of freedom adjustment
    # NOTE: This significantly speeds up repeated calculations where we want
    # to retrieve every possible estimate of feedback uncertainty.
    # NOTE: See linefit() and wiki page for details. Previously thought we only needed
    # to apply reduced dof to t-distribution not standard error (idea was that residual
    # sum in slope standard error formula was incorrectly scaled) but now see that
    # Thompson et al. was correct, since error is *ratio* of two sums over n divided
    # by degrees of freedom, i.e. error should approach zero as samples go to infinity.
    # Similar to uncertainty in population mean estimator versus just raw uncertainty.
    # See: https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
    dof, rsq = {}, covar ** 2 / (dvar * nvar)  # {{{
    idim = 'expand' if dim is None else dim
    keys = [istime if key is None else key for key in np.atleast_1d(correct)]
    keys = ['r' if key and not isinstance(key, str) else key or '' for key in keys]
    with xr.set_options(keep_attrs=True):  # retain numerator units
        resid = numer - (offset + denom * slope)
    if dim is None and any(keys):
        raise ValueError(f'Too many dimensions {dims!r} for {correct=}.')
    if set(keys) - {'x', 'y', 'r', ''}:
        raise ValueError(f"Invalid {correct=}. Must be 'x' 'y' 'r' or ''.")
    for key in keys:
        if key == 'x':
            seq, savg = denom, davg
        elif key == 'y':
            seq, savg = numer, navg
        elif key == 'r':  # see climopy
            seq, savg = resid, (wgts * resid).sum(dims, skipna=True)
        else:  # no-op
            seq, savg = slope.expand_dims(idim).isel({idim: slice(0)}), 0
        if seq.size:  # i.e. non-none correction
            seq = detrend_dims(seq, dim=idim, manual=manual, correct=False)
        anom, ianom = seq - savg, seq.shift({idim: 1}) - savg
        icov = (anom * ianom).mean(idim, skipna=True)
        ivar = (anom ** 2).mean(idim, skipna=True)
        corr = (icov / ivar).fillna(0).clip(0, 1)
        cnt = size * (1 - corr) / (1 + corr) - 2
        cnt = xr.DataArray(cnt, attrs={'units': ''})
        dof[key] = cnt.clip(1, None)
    coord = xr.DataArray([*dof], dims='correct')
    dof = xr.concat(dof.values(), coord)
    dof = dof.squeeze('correct', drop=True) if dof.correct.size == 1 else dof

    # Get standard error and fit terms. Sort if data is one-dimensional
    # NOTE: Arrived at replacement for 1 / n term from wikipedia formula empirically
    # by noting that average x variance ratio term was exactly equivalent to 1 / n.
    # NOTE: This uses a special t-statistic to infer correlation uncertainty bounds.
    # See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Standard_error
    idims = tuple(getattr(coords, 'dims', denom.dims))  # {{{
    fits, bnds, sigmas = [], [], []  # store individual bounds
    isel, dsel, nsel = {}, denom, numer
    if set(dims) - set(idims):
        raise ValueError(f'Invalid input coord dims {idims}. Should include {dims}.')
    if coords is not None:  # preserves coordinates
        dsel = xr.DataArray(coords, name=denom.name, dims=idims)
        dsel.coords.update({} if denom.ndim > 1 else {denom.name: dsel})
        dsel = to_decimal(dsel, coords=False) if istime else dsel
        dsel = dsel.climo.dequantify()
        dsel.attrs.setdefault('units', denom.units)
        dsel = dsel.climo.to_units(denom.units)
        isel, nsel = {}, xr.full_like(dsel, np.nan, dtype=np.float64)
    elif not nosort:
        isel = {dim: np.argsort(denom.values, axis=denom.dims.index(dim))}
        dsel, nsel = denom.isel(isel), numer.isel(isel)
    if not nofit:  # calculate fit and residual
        fit = offset + dsel * slope
        fits.append(fit)
    if not nobnds:  # bounds
        if stat in ('corr', 'covar'):  # see correlation coefficient wiki
            corr = covar / (dstd * nstd)
            sigma = np.sqrt((1 - corr ** 2) / dof)  # correlation error
            sigmas.append((corr / sigma, sigma))
        else:  # see simple linear regression wiki
            rvar = (wgts * resid ** 2).sum(dims, skipna=True)
            sigma = np.sqrt(rvar / dvar / dof)  # regression units
            sigmas.append((slope, sigma))
        if not nofit and not nofitbnds:  # NOTE: false if stat not slope
            rvar = (dsel - davg) ** 2 + (wgts * danom ** 2).sum(dims, skipna=True)
            sigma = np.sqrt(nvar * rvar / dvar / size)  # numerator units
            sigmas.append((fit, sigma))

    # Create bounds and format results
    # WARNING: Critical to include 'nobnds' for case where we use nobnds=True to
    # get raw unsorted best fit and subsequently apply to other data in workflow.
    # WARNING: Critical that 'danoms' and 'dvars' both use same day-per-month weights
    # so that they cancel below. Wikipedia formula just does raw sums.
    for idx, (base, sigma) in enumerate(sigmas):  # {{{
        idof, isigma = xr.broadcast(dof, sigma)  # prepend 'correct'
        datas = fits if idx else bnds
        deltas = var._dist_bounds(isigma, pctile, dof=idof, symmetric=True)
        for delta in deltas:  # lower upper bounds
            dim = ('pctile',) if delta.ndim > isigma.ndim else ()
            bnd = base + xr.DataArray(delta, dims=(*dim, *isigma.dims))
            if stat in ('corr', 'covar'):
                bnd = bnd / np.sqrt(idof + bnd ** 2)  # see wiki page
            if stat == 'covar':
                bnd = bnd * nstd * dstd
            if stat == 'proj':
                bnd = bnd * dstd
            bnd = bnd.transpose(*dim, ..., *isigma.dims)
            bnd.attrs.update(data.attrs)
            datas.append(bnd)
    rsq = covar ** 2 / (dvar * nvar)
    coord = {}
    for fit in fits:
        fit.attrs.update({'units': numer.units})
        fit.coords.update({**coord, 'ymean': navg, 'xdata': dsel, 'ydata': nsel})
    return (data, *bnds, *fits, rsq, dof)
