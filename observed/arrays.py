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
    'standardize_dims',
]

# Message cache
# NOTE: Used to prevent duplicate trend messages
VERBOSE_MESSAGES = set()


def _get_mask(data, ocean=True):
    """
    Return land mask using `global_land_mask`.

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
    lat = data.lat.values
    lon = data.lon.values % 180  # convention
    lon, lat = np.meshgrid(lon, lat)  # creates lat x lon arrays
    if ocean:  # ocean only
        mask = globe.is_ocean(lat, lon)
    else:  # land only
        mask = globe.is_land(lat, lon)
    coords = {'lat': data.lat, 'lon': data.lon}
    mask = xr.DataArray(mask, dims=('lat', 'lon'), coords=coords)
    return mask


def to_decimal(data, dim=None):
    """
    Convert datetime index to decimal year.

    Parameters
    ----------
    data : xarray.DataArray
        The input data with datetime coordinate.
    dim : str, default: 'time'
        The datetime coordinate name.

    Returns
    -------
    result : xarray.DataArray
        The array with decimal year coordinate.
    """
    # TODO: Consider moving this to 'climopy' or 'coupled' (more relevant for models).
    # NOTE: Xarray cftime has 'dayofyear' 'days_in_month' and 'is_leap_year' but no
    # 'days_in_month' so derive from calendar: https://unidata.github.io/cftime/api.html
    dim = dim or 'time'
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
    data = year if data.name == date.name else data
    data = data.assign_coords({dim: year})
    return data


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
    name = data.name
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
    with warnings.catch_warnings():
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
        Whether to print info on the detrend slope.
    **kwargs
        Passed to `regress_dims`.

    Returns
    -------
    result : xarray.DataArray
        The detrended data.
    """
    # Get trends
    # WARNING: Critical to have regress_dims auto-drop 'x', 'y', 'fit',
    # 'fit_lower', and 'fit_upper' coordinates left over from detrending.
    # NOTE: For regional data this will remove regional trends. Then when taking
    # global average will reproduce removal of global average trend.
    dim = dim or 'time'  # intentional error
    base = getattr(data, dim)  # possibly integer
    kwargs = {**kwargs, 'nofit': False}  # always override
    result = regress_dims(base, data, **kwargs)
    beta, _, _, fit, fit_lower, fit_upper, rsq, dof = result
    beta = beta.squeeze(drop=True)
    name = data.name
    if verbose:  # print message
        if 'lon' in beta.sizes and 'lat' in beta.sizes:
            beta = beta.climo.add_cell_measures()
            beta = beta.climo.average('area')
        if 'lon' in dof.sizes and 'lat' in dof.sizes:
            dof = dof.climo.add_cell_measures()
            dof = dof.climo.average('area').item()  # 'units' can now be missing
        trend = f'{10 * beta.item(): >5.2f}'
        units = data.attrs.get('units', None)
        header = name or 'unknown'
        message = f'{header: >8s} trend: {trend} {units} / decade (dof {dof:.0f})'
        if message not in VERBOSE_MESSAGES:
            print(message)
        VERBOSE_MESSAGES.add(message)
    drop = ('x', 'y', 'base')
    base = fit.base.drop_vars('base')
    fit = fit.drop_vars(drop)
    fit_lower = fit_lower.drop_vars(drop)
    fit_upper = fit_upper.drop_vars(drop)
    result = data - fit + base  # WARNING: must come after drop
    result.name = name
    result.attrs.update(data.attrs)
    result.coords['fit'] = fit
    result.coords['fit_lower'] = fit_lower
    result.coords['fit_upper'] = fit_upper
    return result


def regress_dims(
    denom, numer, dim=None, raw=False, stat=None, ocean=None, weights=None,
    coords=None, correct=None, pctile=True, manual=False, nobnds=False, nofit=None,
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
    correct : bool or str, optional
        Whether to correct the standard error for reduced degrees of freedom due
        to serial correlation. If sequence then ``correct`` coordinate is added.
    pctile : bool or float, optional
        Whether to return percentiles bounds or standard error deviations. If float
        this is the percentile range. If sequence then ``pctile`` coordinate is added.
    manual : bool, optional
        Whether to skip automatic weighting by cell measures (e.g. duration) and infer
        degrees of freedom from input weights (if provided) instead of sample count.
    nobnds : bool, optional
        Whether to skip calculating statistic bounds. If ``True`` only the
        first argument is returned.
    nofit : bool, optional
        Whether to skip calculating fit bounds. If ``True`` only four arguments
        are returned. Must be ``False`` if `corr` or `covar` is ``True``.

    Returns
    -------
    result : xarray.DataArray
        The calculated statistic.
    result_lower, result_upper : xarray.dataArray, optional
        The statistic uncertainty bounds. Skipped if `nobnds` is ``True``.
    fit, fit_lower, fit_upper : xarray.DataArray, optional
        The regression fit and uncertainty bounds. Skipped if `nofit` is ``True``.
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
    kw_dims = dict(include_scalar=False, include_pseudo=False)
    nofit = bool(stat) if nofit is None else nofit
    nofit = nofit or nobnds  # always siable if nobnds=True
    denom = denom.climo.dequantify()  # ensure units attribute
    numer = numer.climo.dequantify()  # ensure units attribute
    dims = denom.climo._parse_dims(dim or 'time', **kw_dims)  # must be in denominator!
    dim = dims[0] if len(dims) == 1 else None  # single coordinate
    time = denom.coords['time'] if dim == 'time' and denom.name == 'time' else None
    if stat and not nofit:
        raise ValueError(f'Incompatible arguments {stat=} {nofit=}.')
    if manual:  # input weights only not infer weights
        wgts = xr.ones_like(denom) if weights is None else weights
    else:  # infer automatically (_parse_weights raises error if missing)
        numer = numer.climo.add_cell_measures()
        dims, *parts = numer.climo._parse_weights(dims, weight=weights, **kw_dims)
        wgts = math.prod(wgt.climo.dequantify() for wgts in parts for wgt in wgts)
    if ocean is not None:  # TODO: improve this
        mask1 = _get_mask(denom, ocean=ocean)
        mask2 = _get_mask(numer, ocean=ocean)
        wgts = xr.where(mask1 | mask2, wgts, 0)
    if time is not None:  # WARNING: must be called after add_cell_measures()
        denom = to_decimal(denom).astype(numer.dtype)
        numer = to_decimal(numer)
        wgts = to_decimal(wgts)
    denom = denom.drop_vars(denom.coords.keys() - denom.sizes.keys())
    numer = numer.drop_vars(numer.coords.keys() - numer.sizes.keys())
    wgts = wgts.drop_vars(wgts.coords.keys() - wgts.sizes.keys())
    denom, numer, wgts = xr.align(denom, numer, wgts)  # align 'dims' coordinates
    denom, numer, wgts = xr.broadcast(denom, numer, wgts)  # broadcast other coordinates
    nulls = numer.isnull() | denom.isnull()
    numer, denom, wgts = numer.where(~nulls), denom.where(~nulls), wgts.where(~nulls)

    # Calculate statistic
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
    if manual:  # degrees of freedom from weights
        size = wgts.sum(dims)
    else:  # normalize and get size
        size = math.prod(numer.sizes[key] for key in dims)
    wgts = wgts / wgts.sum(dims)  # now (w * d).sum() is effectively d.sum() / n
    davg = 0 if raw else (wgts * denom).sum(dims, skipna=True)
    navg = 0 if raw else (wgts * numer).sum(dims, skipna=True)
    covar = wgts * (denom - davg) * (numer - navg)
    covar = covar.sum(dims, skipna=True)
    dvar = wgts * (denom - davg) ** 2
    dvar = dvar.sum(dims, skipna=True)
    rsq = None  # skip calculating by default
    slope = covar / dvar
    offset = navg - slope * davg
    if not nobnds or stat == 'corr':
        nvar = wgts * (numer - navg) ** 2
        nvar = nvar.sum(dims, skipna=True)
        rsq = covar ** 2 / (dvar * nvar)
    if not stat or stat == 'slope':
        result = slope
        units = f'{numer.units} / {denom.units}'
    elif stat == 'cov' or stat == 'covar':
        result = covar
        units = f'{numer.units} {denom.units}'
    elif stat == 'proj':
        result = covar / np.sqrt(dvar)
        units = numer.units
    elif stat == 'corr':
        result = covar / (np.sqrt(dvar) * np.sqrt(nvar))
        units = ''
    else:
        raise ValueError(f'Unknown input statistic {stat!r}.')
    result.attrs.update(units=units)

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
    dofs = {}
    for correct in np.atleast_1d(correct):
        correct = time is not None if correct is None else correct
        correct = 'r' if correct is True else correct or ''
        if not correct:
            seq = None
        elif dim is None:
            raise ValueError(f'Too many dimensions {dims!r} for {correct=}.')
        elif correct == 'x':
            seq, savg = denom, davg
        elif correct == 'y':
            seq, savg = numer, navg
        elif correct == 'r':  # see climopy
            with xr.set_options(keep_attrs=True):
                seq = numer - (offset + denom * slope)  # correlate in *time*
                savg = (wgts * seq).sum(dims, skipna=True)
        else:
            raise ValueError(f"Invalid {correct=}. Must be 'x' or 'y'.")
        if seq is not None and time is not None:
            seq = seq.assign_coords({'time': time})
        if seq is None:
            auto = xr.zeros_like(slope)
        else:  # get adjustment factor
            seq = detrend_dims(seq, manual=manual, correct=False)
            seq1 = seq.isel({dim: slice(1, None)})
            assign = {dim: seq1.coords[dim].values}
            seq2 = seq.isel({dim: slice(None, -1)}).assign_coords(assign)
            sigma = ((seq - savg) ** 2).mean(dim, skipna=True)
            auto = ((seq1 - savg) * (seq2 - savg)).mean(dim, skipna=True) / sigma
        auto = np.clip(auto, 0, None)
        scale = (1 - auto) / (1 + auto)  # effective samples
        idof = size * scale - 2  # see above
        idof.attrs['units'] = ''
        dofs[correct] = idof
    if len(dofs) == 1:
        _, dof = dofs.popitem()
    else:
        dof = xr.concat(list(dofs.values()), xr.DataArray(list(dofs), dims='correct'))

    # Get standard error and fit terms. Sort if data is one-dimensional
    # NOTE: Arrived at replacement for 1 / n term from wikipedia formula empirically
    # by noting that average x variance ratio term was exactly equivalent to 1 / n.
    # NOTE: This uses a special t-statistic to infer correlation uncertainty bounds.
    # See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Standard_error
    # WARNING: Critical that 'danoms' and 'dvars' both use same day-per-month weights
    # so that they cancel below. Wikipedia formula just does raw sums.
    # ic((slope_upper - slope_lower) * denom.mean(), (fit_upper - fit_lower).mean())
    # ic(numer.name, denom.name, slope, sigma, slope_lower, slope_upper, dof)
    # ic(fit.mean(), (fit_upper - fit_lower).mean(), fit_upper.mean())
    isel = {}
    bnds, ibnds = [], []  # store individual bounds
    dsort, nsort, wsort = denom, numer, wgts
    if not nofit and dim is not None:  # sort to make plotting easier
        axis = denom.dims.index(dim)
        isel = {dim: np.argsort(denom.values, axis=axis)}
        dsort, nsort, wsort = denom.isel(isel), numer.isel(isel), wgts.isel(isel)
    if not nobnds and stat in ('corr', 'covar'):  # see wikipedia correlation page
        base = np.sqrt(dvar) * np.sqrt(nvar)
        corr = covar / base
        sigma = np.sqrt((1 - corr ** 2) / dof)  # correlation error
        ibnds.append((corr / sigma, sigma))
    elif not nobnds:  # see wikipedia simple linear regression page
        fit = (offset + dsort * slope).transpose(*dims, ...)
        resid = (wsort * (nsort - fit) ** 2).sum(dims, skipna=True)
        sigma = np.sqrt(resid / dvar / dof)  # regression error
        ibnds.append((slope, sigma))
    if not nofit:  # only allowed if not corr and not covar
        danom = (dsort - davg) ** 2 / dvar / size
        danom = danom + (wsort * danom).sum(dims, skipna=True)  # valid after testing
        sigma = np.sqrt(danom * resid)  # approximate
        ibnds.append((fit, sigma))
    for base, sigma in ibnds:
        idof, isigma = xr.broadcast(dof, sigma)  # prepend 'correct'
        deltas = var._dist_bounds(isigma, pctile, dof=idof, symmetric=True)
        for delta in deltas:
            dims = isigma.dims if isigma.ndim == delta.ndim else ('pctile', *isigma.dims)  # noqa: E501
            bound = base + xr.DataArray(delta, dims=dims)
            if stat in ('corr', 'covar'):
                bound = bound / np.sqrt(idof + bound ** 2)  # see wiki page
            if stat == 'covar':
                bound = bound * np.sqrt(nvar) * np.sqrt(dvar)
            if stat == 'proj':
                bound = bound * np.sqrt(dvar)
            bound.attrs.update(result.attrs)
            bnds.append(bound)
    kw_interp = {'method': 'linear', 'kwargs': {'fill_value': 'extrapolate'}}
    fits, ifits = [], () if nofit else (fit, *bnds[2:])
    bnds = () if nobnds else (*bnds[:2],)
    for fit in ifits:
        if coords is None:
            assign = {} if time is None else {'time': time.isel(isel)}
            fit = fit.assign_coords(assign)
            xdata = dsort.assign_coords(assign)
            ydata = nsort.assign_coords(assign)
        elif dim is not None:  # interpolate to input coordinates
            fit = fit.drop_vars(dim).rename({dim: denom.name})
            fit = fit.assign_coords({denom.name: dsort.values})
            fit = fit.interp({denom.name: coords}, **kw_interp)
            fit = fit.drop_vars(denom.name)
            xdata = xr.DataArray(coords, dims=dim)  # ok if coords is already DataArray
            ydata = xr.full_like(coords, np.nan)  # no corresponding 'actual data'
        else:
            raise ValueError(f'Too many dimensions {dims!r} for input fit coordinates.')
        fit.attrs.update(units=numer.units)
        fit.coords.update({'x': xdata, 'y': ydata, 'base': navg})
        fits.append(fit)
    return result if nobnds and nofit else (result, *bnds, *fits, rsq, dof)


def standardize_dims(data):  # noqa: U100
    """
    Standardize data onto the same horizontal grid.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.

    Returns
    -------
    result : xarray.DataArray
        The standardized data.
    """
    # TODO: Translate the following bash code to pycdo. See cmip_data process.py
    # for folder in ceres gistemp4 hadcrut5; do
    #   for file in ~/data/$folder/*.nc; do
    #     output=${file%.nc}_standardized.nc
    #     [[ "$file" =~ global|weights|standardized ]] && continue
    #     [ -r "$output" ] && continue
    #     echo "Remapping file: ${file#$HOME/}"
    #     cdo remapcon,r72x36 "$file" "$output"
    #   done
    # done
