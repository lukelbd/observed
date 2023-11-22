#!/usr/bin/env python3
"""
Observed processing utilities.
"""
import warnings
from icecream import ic  # noqa: F401

import numpy as np  # noqa: F401
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401


__all__ = [
    'annual_average',
    'annual_filter',
    'detrend_series',
    'regress_series',
    'standardize_grid',
]

# Message cache
# NOTE: Used to prevent duplicate trend messages
VERBOSE_MESSAGES = set()


def annual_average(data, **kwargs):
    """
    Return annual averages computed starting from the input month.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.
    **kwargs
        Passed to `annual_filter`.
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
    start = data.time[0].dt.strftime('%b').item().upper()
    with warnings.catch_warnings():  # datetime64 emits annoying warning
        warnings.simplefilter('ignore')
        days = data.time.dt.days_in_month  # do not normalize yet, weight for resample()
        data = data.astype(data.dtype)  # preserve data type
        with xr.set_options(keep_attrs=True):
            numer = (data * days).resample(time=f'AS-{start}')
            denom = (xr.ones_like(data) * days).resample(time=f'AS-{start}')
            data = numer.sum(dim='time', skipna=False) / denom.sum(dim='time')
    data.name = name
    return data


def annual_filter(data, month=None, anomaly=False, **kwargs):
    """
    Return data filtered to include only full years.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.
    month : int, optional
        The starting month for the averages.
    anomaly : bool, optional
        Whether to subtract monthly averages from subselection.
    **kwargs
        Passed to `.sel()`. Use for further subselecttions.
    """
    # NOTE: Critical even for monthly regressions to prevent e.g. residual seasonal
    # component in either the climate or the forced response from influencing results.
    data = data.sel(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        months = data.time.dt.month.values
    if np.unique(months).size == 1:
        return data  # already annual
    if month is not None:
        month = int(month)  # integer month
    else:
        month = data.time[0].dt.month.item()  # integer month
    idxs1, = np.nonzero(months == month)
    idxs2, = np.nonzero(months == (month - 2) % 12 + 1)  # 1 --> 12
    data = data.isel(time=slice(idxs1[0], idxs2[-1] + 1))
    if anomaly:  # note data already in anomaly form, but this uses selected period
        climate = data.groupby('time.month').mean()
        with xr.set_options(keep_attrs=True):
            data = data.groupby('time.month') - climate
        data = data.drop_vars('month')
    return data


def detrend_series(data, verbose=False, **kwargs):
    """
    Return detrended input data.

    Parameters
    ----------
    data : xarray.DataArray
        The input data.
    verbose : bool, optional
        Whether to print info on the detrend slope.
    **kwargs
        Passed to `_regress_series`.
    """
    # Get trends
    # NOTE: For regional data this will remove regional trends. Then when taking
    # global average will reproduce removal of global average trend.
    kwargs.setdefault('adjust', True)
    result = regress_series(data, data.time, **kwargs)
    beta, _, _, dof, fit, fit_lower, fit_upper = result
    beta = beta.squeeze(drop=True)
    name = data.name
    if verbose:  # print message
        if 'lon' in beta.sizes and 'lat' in beta.sizes:
            beta = beta.climo.add_cell_measures()
            beta = beta.climo.average('area')
        if isinstance(dof, xr.DataArray) and 'lon' in dof.sizes and 'lat' in dof.sizes:
            dof.attrs.update(units='')
            dof = dof.climo.add_cell_measures()
            dof = np.round(dof.climo.average('area')).astype(int).item()
        trend = f'{10 * beta.item(): >5.2f}'
        units = data.attrs.get('units', None)
        header = name or 'unknown'
        message = f'{header: >8s} trend: {trend} {units} / decade (dof {dof})'
        if message not in VERBOSE_MESSAGES:
            print(message)
        VERBOSE_MESSAGES.add(message)

    # Update averages
    # WARNING: Critical to have _regress_series auto-drop 'x', 'y', 'fit',
    # 'fit_lower', and 'fit_upper' coordinates left over from detrending.
    with warnings.catch_warnings():  # datetime64 emits annoying error
        warnings.simplefilter('ignore')
        wgts = fit.time.dt.days_in_month
        wgts = wgts / wgts.sum('time')
    favg = (wgts * fit).sum('time', skipna=False)
    anoms = data - fit + favg
    fit = fit.drop_vars(('x', 'y'))
    fit_lower = fit_lower.drop_vars(('x', 'y'))
    fit_upper = fit_upper.drop_vars(('x', 'y'))
    anoms.name = name
    anoms.attrs.update(data.attrs)
    anoms.coords['fit'] = fit
    anoms.coords['fit_lower'] = fit_lower
    anoms.coords['fit_upper'] = fit_upper
    return anoms


def regress_series(numer, denom, coords=None, noweight=False, pctile=True, adjust=True):  # noqa: E501
    """
    Return regression parameters.

    Parameters
    ----------
    numer, denom : xarray.DataArray
        The numerator and denominator.
    coords : array-like, optional
        The coordinates used for the fit. Default is `denom`.
    noweight : bool, optional
        Whether to skip weighting by days-per-month.
    pctile : bool or float, optional
        Whether to use percentiles or standard errors. If float this is the %% range.
    adjust : bool or str, optional
        Whether to adjust for degrees of freedom. If string this specifies which.
    """
    # Get regression estimates
    # WARNING: Critical to auto-drop 'x', 'y', 'fit', 'fit_lower', and 'fit_upper'
    # coordinates left over from detrending or else get weird hard-to-debug errors.
    # NOTE: For details see coupled.process._components_slope (uses climo.linefit) and
    # cmip_data.feedbacks._regress_monthly. Combines 'linefit' behavior returning extra
    # uncertainty parameters and '_regress_monthly' weighting behavior plus fixes
    # Thompson et al. standard error issue. Should add to climopy in future
    if denom.name == 'time':  # assume identical spacing
        denom = xr.ones_like(denom, numer.dtype)
        with warnings.catch_warnings():  # datetime64 emits annoying warning
            warnings.simplefilter('ignore')
            months = denom.time.dt.month
        denom = denom.cumsum('time') / len(np.unique(months))
    numer = numer.drop_vars(numer.coords.keys() - numer.sizes.keys())
    denom = denom.drop_vars(denom.coords.keys() - denom.sizes.keys())
    numer, denom = xr.align(numer, denom)
    with warnings.catch_warnings():  # datetime64 emits annoying warning
        warnings.simplefilter('ignore')
        if noweight:
            wgts = xr.ones_like(denom)
        else:  # no-op if already annual averaged
            wgts = denom.time.dt.days_in_month  # after 'annual_filter'
    wgts = wgts / wgts.sum('time')  # note now (w * d).sum() is effectively d.sum() / n
    navg = (wgts * numer).sum('time', skipna=True)
    davg = (wgts * denom).sum('time', skipna=True)
    covar = wgts * (denom - davg) * (numer - navg)
    covar = covar.sum('time', skipna=True)
    dvar = wgts * (denom - davg) ** 2
    dvar = dvar.sum('time', skipna=True)
    slope = covar / dvar
    offset = navg - slope * davg
    slope.attrs.update(units=f'{numer.units} / year')
    offset.attrs.update(units=numer.units)

    # Get degrees of freedom adjustment
    # NOTE: See linefit() and wiki page for details. Previously thought we only needed
    # to apply reduced dof to t-distribution not standard error (idea was that residual
    # sum in slope standard error formula was incorrectly scaled) but now see that
    # Thompson et al. was correct, since error is *ratio* of two sums over n divided
    # by degrees of freedom, i.e. error should approach zero as samples go to infinity.
    # Similar to uncertainty in population mean estimator versus just raw uncertainty.
    # See: https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
    if not adjust:  # no adjustment
        factor = 1
        dof = numer.time.size - 2
    else:  # serial correlation
        if not isinstance(adjust, str):
            seq = numer - (offset + denom * slope)  # correlate in *time*
            savg = (wgts * seq).sum('time', skipna=True)
        elif adjust == 'y':
            seq = numer  # correlate in *time*
            savg = navg
        elif adjust == 'x':
            seq = denom  # correlate in *time*
            savg = davg
        else:
            raise ValueError(f"Invalid {adjust=}. Must be 'x' or 'y'.")
        seq1 = seq.isel(time=slice(1, None))
        seq2 = seq.isel(time=slice(None, -1))
        seq2 = seq2.assign_coords(time=seq1.time.values)  # avoid alignment!
        scale = ((seq - savg) ** 2).mean('time', skipna=True)
        autocov = ((seq1 - savg) * (seq2 - savg)).mean('time', skipna=True)
        autocorr = np.clip(autocov / scale, 0, None)
        factor = ((1 - autocorr) / (1 + autocorr))  # effective samples
        dof = numer.time.size * factor - 2  # see above
        dof = np.round(dof).astype(np.int64)
        dof = dof.item() if dof.size == 1 else dof

    # Get standard error and fit terms
    # NOTE: See process.py _components_slope() for details. Here do not need 'pctile'
    # dimension on either slope uncertainty range or best fit uncertainty range.
    # NOTE: Arrived at replacement for 1 / n term from wikipedia formula empirically
    # by noting that average x variance ratio term was exactly equivalent to 1 / n.
    # WARNING: Critical that 'danoms' and 'dvars' both use same day-per-month weights
    # so that they cancel below. Wikipedia formula just does raw sums.
    axis = denom.dims.index('time')
    isel = {'time': np.argsort(denom.values, axis=axis)}
    nsort, dsort, wsort = numer.isel(isel), denom.isel(isel), wgts.isel(isel)
    fit = offset + dsort * slope
    fit = fit.transpose('time', ...)
    fit.attrs.update(units=numer.units)
    resid = (nsort - fit) ** 2
    resid = (wsort * resid).sum('time', skipna=True)
    sigma = np.sqrt(resid / dvar / dof)
    danom = wsort * (dsort - davg) ** 2 / dvar
    danom = danom + danom.mean('time', skipna=True)  # tested
    with xr.set_options(keep_attrs=True):
        fit_sigma = np.sqrt(danom * resid)  # approximate
        del_lower, del_upper = var._dist_bounds(fit_sigma, pctile, dof=dof)
        fit_lower, fit_upper = fit + del_lower, fit + del_upper
        sigma_lower, sigma_upper = var._dist_bounds(sigma, pctile, dof=dof)
        slope_lower, slope_upper = slope + sigma_lower, slope + sigma_upper
    if coords is not None:  # should be equally spaced
        fits = []
        for part in (fit, fit_lower, fit_upper):
            part = part.drop_vars('time').rename(time='denom')
            part = part.assign_coords(denom=dsort.values)
            part = part.interp(denom=coords, method='linear', kwargs={'fill_value': 'extrapolate'})  # noqa: E501
            part = part.drop_vars('denom').rename(denom='time')
            fits.append(part)
        fit, fit_lower, fit_upper = fits
        dsort = xr.DataArray(coords, dims='time')
        nsort = xr.full_like(dsort, np.nan)
    fit.coords.update({'x': dsort, 'y': nsort})
    fit_lower.coords.update({'x': dsort, 'y': nsort})
    fit_upper.coords.update({'x': dsort, 'y': nsort})
    # ic((slope_upper - slope_lower) * denom.mean(), (fit_upper - fit_lower).mean())
    # ic(numer.name, denom.name, slope, sigma, slope_lower, slope_upper, dof)
    # ic(fit.mean(), (fit_upper - fit_lower).mean(), fit_upper.mean())
    return slope, slope_lower, slope_upper, dof, fit, fit_lower, fit_upper


def standardize_grid(data):
    """
    Standardize data onto the same horizontal grid.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        The input data.
    """
    # TODO: Translate the following bash code to pycdo. See cmip_data process.py
    # for folder in ceres gistemp4 hadcrut5; do
    #   for file in ~/data/$folder/*.nc; do
    #     [[ "$file" =~ global|weights ]] && continue
    #     echo "Remapping file: ${file#$HOME/}"
    #     cdo remapcon,r72x36 "$file" "${file%.nc}_standardized.nc"
    #   done
    # done
    data
