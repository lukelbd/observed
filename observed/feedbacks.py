#!/usr/bin/env python3
"""
Get observational forcing-feedback estimates.
"""
import itertools
import os
from datetime import datetime
from pathlib import Path
from icecream import ic  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401

from .datasets import load_dataset
from .arrays import annual_average, annual_filter, detrend_dims, regress_dims

__all__ = ['calc_feedback', 'process_spatial', 'process_scalar']

# Radiative flux components
LABELS_CLOUD = {
    '': 'all-sky',
    'cs': 'clear-sky',
    'ce': 'cloud',
    'cld': 'cloud',
}
LABELS_WAVELEN = {
    'f': 'net',
    'l': 'longwave',
    's': 'shortwave',
    'u': 'reflected',
    'd': 'insolation',
}

# Temperature source labels
LABELS_SOURCE = {
    'he': 'He et al.',
    'gis': 'GISTEMP4',
    'had': 'HadCRUT5',
}
LONGS_SOURCE = {
    'he': 'external',
    'gis': 'gistemp',
    'had': 'hadcrut',
}


# Feedback default arguments
PARTS_DEFAULT = {
    'wav': ('f', 'l', 's'),  # separate variable
    'sky': ('', 'cs', 'ce', 'cld'),  # separate variable
    'cld': ('',),  # instead use 'sky'
}
PARTS_TESTING = {key: value[:2] for key, value in PARTS_DEFAULT.items()}
PARAMS_DEFAULT = {
    'source': ('gis', 'had'),
    'years': (None,),  # named 'period'
    'month': ('jan', 'jul'),  # named 'initial'
    'annual': (False, True),  # named 'style'
    'anomaly': (True, False),  # named 'remove'
    'detrend': ('xy', 'x', 'y', ''),
    'correct': ('r', 'y', 'x', ''),
}
PARAMS_TESTING = {key: value[:2] for key, value in PARAMS_DEFAULT.items()}

# Parameter coordinate translations
TRANSLATE_PARAMS = {
    ('annual', None): ('style', 'monthly'),
    ('annual', False): ('style', 'monthly'),
    ('annual', True): ('style', 'annual'),
    ('anomaly', None): ('remove', 'climate'),
    ('anomaly', False): ('remove', 'climate'),
    ('anomaly', True): ('remove', 'average'),
    ('internal', None): ('error', 'regression'),
    ('internal', False): ('error', 'regression'),
    ('internal', True): ('error', 'internal'),
}
TRANSLATE_PARAMS.update(  # _parse_coords sets 'None'
    {('years', True): ('period', '20yr'), ('month', None): ('initial', 'init')},
)
TRANSLATE_PARAMS.update(  # _parse_coords sets 'None'
    {('month', n): ('initial', datetime(1, n, 1).strftime('%b')) for n in range(1, 13)},
)


def _parse_names(source=None, wav=None, sky=None, cld=None, sfc=None):
    """
    Return the part names and labels for the input feedback specification.

    Parameters
    ----------
    source : str or sequence, default: 'had'
        The temperature data source. One of ``'had'``, ``'gis'``, or ``'he'``.
    wav : str or sequence, default: ('f', 's', 'l')
        The flux wavelength. One of ``'f'``, ``'s'``, ``'l'``.
    cld : bool or str, optional
        Whether to use masking-adjusted cloud feedbacks from He et al. paper.
    sky : str, default: ''
        The flux sky component. One of ``''``, ``'cs'``, ``'ce'``.
    sfc : str, default: 't'
        The flux boundary. One of ``'t'``, ``'s'``, ``'a'``.

    Returns
    -------
    temps, fluxes : list of str
        The temperature and flux variable(s).
    templabels, fluxlabels : list of str, optional
        The temperature and flux readable labels.
    pathlabel : str
        The path label suitable for figures.
    """
    sources = ('gis', 'had') if source is None else source  # pass False or () to skip
    sources = () if not sources else (source,) if isinstance(source, str) else sources
    wavs = ('f', 's', 'l') if wav is None else wav  # pass False or () to skip
    wavs = () if not wavs else (wav,) if isinstance(wav, str) else wavs
    cld = 'cld' if cld is True else cld
    sky = '' if sky is None else sky
    sfc = 't' if sfc is None else sfc
    temps = tuple(f'ts_{source}' for source in sources)
    names = tuple(f'rs{wav}{sfc}' if wav in 'ud' else f'r{wav}n{sfc}' for wav in wavs)  # noqa: E501
    fluxes = tuple(f'{cld}_{part}' if cld else f'{part}{sky}' for part in names)
    prefix = 'ts-' + '-'.join(sources)
    suffix = cld if cld is not None else sky
    templabels = [LABELS_SOURCE[source] for source in sources]
    fluxlabels = [f'{LABELS_WAVELEN[wav]}' for wav in wavs]
    pathlabel = '_'.join(filter(None, (prefix, '-'.join(names), suffix)))
    return temps, fluxes, templabels, fluxlabels, pathlabel


def _parse_coords(time, translate=None, **kwargs):
    """
    Parse `calc_feedback` parameters into feedback version coordinates.

    Parameters
    ----------
    time : xarray.DataArray
        The reference time coordinates.
    translate : dict, optional
        Additional translations from ``(name, value)`` pairs to coordinates.
    **kwargs
        The `calc_feedback` keyword arguments.

    Returns
    -------
    coords : dict
        The feedback version coordinates.
    """
    # NOTE: If e.g. have starting month 'jan' for CERES data running from 'mar' to
    # 'nov' then version label will still be e.g. '23yr' even though annual_filter()
    # restricts record to 22 years (i.e. full 12-month blocks). Keep for consistency.
    month = time[0].dt.strftime('%b').item()
    year0, year1 = time[0].dt.year.item(), time[-1].dt.year.item()
    translate = {**TRANSLATE_PARAMS, **(translate or {})}
    translate.setdefault(('month', None), ('initial', month.lower()))
    translate.setdefault(('years', None), ('period', f'{year0}-{year1}'))
    coords = {}
    for key, value in kwargs.items():
        if key in ('pctile', 'correct'):  # concatenated by calc_feedback
            continue
        if (key, None) in translate:  # name translation
            name, _ = translate[key, None]
        else:  # e.g. 'source' and 'detrend'
            name = key
        if key == 'source':
            value = LONGS_SOURCE.get(value, value)  # e.g. 'eraint' models
        if isinstance(value, slice):
            value = (value.start, value.stop)
        if np.iterable(value) and not isinstance(value, str):
            value = tuple(value)  # hashable
        if (key, value) in translate:
            _, value = translate[key, value]
        elif key == 'years' and np.isscalar(value):
            value = f'{np.array(value).item()}yr'
        else:  # fallback version label
            value = '-'.join(map(str, np.atleast_1d(value).tolist()))
        coords[name] = value
    return coords


def _parse_kwargs(skip_keys=None, skip_values=None, testing=None, **kwargs):
    """
    Return standardized keyword arguments for the feedback estimate.

    Parameters
    ----------
    skip_keys : str or tuple, optional
        The keys to ignore when parsing.
    skip_values : str or tuple, optional
        The values to ignore when parsing.
    testing : bool, optional
        Whether to use faster default options.
    **kwargs
        Passed to `_parse_names` or `calc_feedback`.

    Returns
    -------
    params : dict
        The vectorized keyword arguments `source`, `years`, `month`, `annual`,
        `anomaly`, `detrend`, `correct` (translated into `version`).
    parts : dict
        The vectorized keyword arguments `wav`, `sky`, and `cld`.
    kwargs : dict
        The additional keyword arguments.
    """
    names = kwargs.pop('name', None)
    skip_keys = (skip_keys,) if isinstance(skip_keys, str) else tuple(skip_keys or ())
    skip_values = (skip_values,) if isinstance(skip_values, str) else tuple(skip_values or ())  # noqa: E501
    if names is None:
        defaults = PARTS_TESTING.copy() if testing else PARTS_DEFAULT.copy()
        defaults = {
            key: tuple(value for value in values if value not in skip_values)
            for key, values in defaults.items() if key not in skip_keys
        }
        parts = {key: kwargs.pop(key, value) for key, value in defaults.items()}
        parts = {key: value if isinstance(value, (list, tuple)) else (value,) for key, value in parts.items()}  # noqa: E501
    elif label := ', '.join(f'{key}={value!r}' for key, value in kwargs.items() if key in PARTS_DEFAULT and value is not None):  # noqa: E501
        raise TypeError(f'Keyword arguments {label} incompatible with {names}={names!r}.')  # noqa: E501
    else:
        parts = {'name': (names,) if isinstance(names, str) else tuple(names)}
    defaults = PARAMS_TESTING.copy() if testing else PARAMS_DEFAULT.copy()
    defaults = {
        key: tuple(value for value in values if value not in skip_values)
        for key, values in defaults.items() if key not in skip_keys
    }
    params = {key: kwargs.pop(key, value) for key, value in defaults.items()}
    params = {key: value if isinstance(value, (list, tuple)) else (value,) for key, value in params.items()}  # noqa: E501
    correct = params.pop('correct', True)  # regress_dims() creates coordinate
    if 'correct' not in skip_keys:
        kwargs.setdefault('correct', correct)
    return params, parts, kwargs


def calc_feedback(
    dataset, *args, source=None, wav=None, sky=None, cld=None, sfc=None,
    years=None, annual=False, detrend=False, verbose=False, **kwargs,
):
    """
    Return climate feedback calculation for global time series.

    Parameters
    ----------
    dataset : xarray.Dataset
        The flux and temperature data from `open_dataset`.
    *args : str, optional
        The optional explicit variable names.
    years : int or 2-tuple, optional
        The number of years to sample for internal variability or the year range.
    annual : bool, optional
        Whether to use annual anomalies instead of monthly anomalies.
    detrend : bool or str, default: False
        Whether to detrend the data. Use ``'x'`` for temperature and ``'y'`` for flux.
    verbose : bool or int, optional
        Whether to print a message after detrending. ``1`` is ``R``, ``2`` is ``T``.
    source, wav, sky, cld, sfc : optional
        Passed to `_parse_names`.
    **kw_annual
        Passed to `annual_average` or `annual_filter`.
    **kw_regress
        Passed to the `regress_dims`.

    Returns
    -------
    lam, lam_lower, lam_upper : xarray.DataArray
        The feedbacks for the input settings.
    fit, fit_lower, fit_upper : xarray.DataArray
        The Gregory regresson fit for the input settings.
    """
    # Get the data arrays
    # NOTE: Here 'years' is used to both specify period size when bootstrapping control
    # or historical data or to restrict time range when passed to `annual_filter`.
    # NOTE: Critical to pass time=None or else time average taken automatically. Note
    # flux components e.g. 'rfnt' will be auto-derived from e.g. 'rsdt' 'rsut' 'rlut'.
    from coupled.process import get_result
    from coupled.specs import _pop_kwargs
    kw_annual = _pop_kwargs(kwargs, annual_filter)
    kw_regress = kwargs.copy()
    kw_detrend = {**kwargs, 'correct': False}
    kw_input = dict(source=source, wav=wav, sky=sky, cld=cld, sfc=sfc)
    kw_input = {key: value for key, value in kw_input.items() if value is not None}
    kw_parts = kw_input.copy()
    nofit = kw_detrend.pop('nofit', False)  # remove from 'detrend'
    for key, value in (('wav', 'f'), ('source', 'gis')):
        kw_parts.setdefault(key, value)
    if not np.isscalar(years):
        kw_annual['years'], years = years, None
    if not args:
        (temp,), (flux,), *_ = _parse_names(**kw_parts)
    elif len(args) != 2:
        raise TypeError(f'Got {len(args)} positional names but expected zero or two.')
    elif label := ', '.join(f'{key}={value!r}' for key, value in kw_input.items()):
        raise TypeError(f'Keyword arguments {label} incompatible with positional arguments.')  # noqa: E501
    else:  # user input values
        temp, flux = args
    if detrend in ('xy', 'yx') or detrend is True:
        idxs = (1, 0)
    elif detrend == 'x':  # temp only
        idxs = (1,)
    elif detrend == 'y':  # flux only
        idxs = (0,)
    elif not detrend:
        idxs = ()
    else:
        raise ValueError(f'Invalid {detrend=}.')
    temp = get_result(dataset, temp, time=None)  # forget time coordinates
    flux = get_result(dataset, flux, time=None)

    # Get the feedbacks and combine results
    # WARNING: If calculating e.g. feedbacks starting in July for 150-year control
    # simualtion annual_filter() returns 149 full years, so when selecting sub-periods
    # allow the final period to be up to 12 months shorter due to truncation.
    # NOTE: Previously embedded annual stuff inside regress_dims so **kwargs would
    # do it twice but with bootstrapping need to ensure correct starting months are
    # selected or e.g. might select 19-year sample instead of 20-year sample.
    mask = ~temp.isnull() & ~flux.isnull()
    temp, flux = temp[mask], flux[mask]
    func = annual_average if annual else annual_filter
    temp, flux = func(temp, **kw_annual), func(flux, **kw_annual)
    coords = None  # fit coordinates
    times = [0]  # regression index
    scale = 1 if annual else 12
    size = flux.size  # regression size
    lams, lams_lower, lams_upper, dofs = [], [], [], []
    fits, fits_lower, fits_upper = [], [], []
    if years:
        size = 20 if years is True else years
        coords = np.linspace(np.min(temp), np.max(temp), 100)
        if size % 2:
            raise ValueError(f'Internal variability years {years} must be even.')
        size, step = size * scale, size * scale // 2
        times = np.arange(0, flux.size - size + scale, step)  # NOTE: see above
    for time in times:
        datas = []
        for idx, data in enumerate((temp, flux)):
            data = data[time:time + size]
            if idx in idxs:  # note detrend
                verb = True if verbose and time == times[0] else False
                data = detrend_dims(data, verbose=verb, **kw_detrend)
            datas.append(data)
        result = regress_dims(*datas, coords=coords, **kw_regress)
        lam, lam_lower, lam_upper, *_, dof = result
        lams.append(lam)
        lams_lower.append(lam_lower)
        lams_upper.append(lam_upper)
        dofs.append(dof)
        if not nofit:
            _, _, _, fit, fit_lower, fit_upper, *_ = result
            fits.append(fit)
            fits_lower.append(fit_lower)
            fits_upper.append(fit_upper)

    # Combine results
    # NOTE: Use percentiles informed from standard deviation normal distribution
    # instead of ensemble since needed for figures.py variability constraint methodology
    # TODO: Consider alternative method from Dessler + Forster randomly sampling
    # points from full time series instead of contiguous segements.
    # TODO: Consider scipy stats-style model class objects that calculate
    # and return different products upon request (e.g. fits and percentiles).
    kw_concat = dict(dim='sample', coords='minimal', compat='override')
    if years:
        lam = xr.concat(lams, **kw_concat)
        lam_lower = xr.concat(lams_lower, **kw_concat)
        lam_upper = xr.concat(lams_upper, **kw_concat)
        dof = xr.concat(dofs, **kw_concat)
    lam_name = f'{flux.name}_lam'
    lam.name = lam_name
    lam_lower.name = f'{lam_name}_lower'
    lam_upper.name = f'{lam_name}_upper'
    result = (lam, lam_lower, lam_upper, dof)
    if not nofit:  # fit processing
        coords = dict(fits[0].coords)  # standardized dependent coordinates
        if years:
            fit = xr.concat(fits, **kw_concat).assign_coords(coords)
            fit_lower = xr.concat(fits_lower, **kw_concat).assign_coords(coords)
            fit_upper = xr.concat(fits_upper, **kw_concat).assign_coords(coords)
        fit_name = f'{flux.name}_fit'
        fit.name = fit_name
        fit_lower.name = f'{fit_name}_lower'
        fit_upper.name = f'{fit_name}_lower'
        result = (*result, fit, fit_lower, fit_upper)
    return result


def process_spatial(dataset=None, output=None, **kwargs):
    """
    Save climate feedbacks using `cmip_data.process` and `cmip_data.feedbacks`.

    Parameters
    ----------
    dataset : xarray.Dataset, optional
        The input dataset. Default is ``load_dataset(globe=True)``.
    output : path-like, optional
        The output directory or name. If ``False`` nothing is saved.
    **kwargs
        Passed to `annual_filter` and `_feedbacks_from_fluxes`.
    """
    # Initial stuff
    # NOTE: _feedbacks_from_fluxes() does not support toggling detrending on-off so
    # instead detrend temperature below. Then feedbacks normalized by local T still
    # capture 'internal' component and when _feedbacks_from_fluxes() takes global
    # average the 'globe' region contains detrended global average. See jupyter
    # notebook for comparison of average process_spatial() with process_scalar().
    from cmip_data.feedbacks import _feedbacks_from_fluxes
    from coupled.process import get_result
    if dataset is None:  # load local observations
        dataset = load_dataset(globe=False)
    if not isinstance(dataset, xr.Dataset):
        raise ValueError('Input argument must be a dataset.')
    if 'correct' in kwargs or 'pctile' in kwargs:
        raise TypeError('Invalid input arguments.')
    skip_keys, skip_values = ('correct', 'detrend'), ('f', 'ce', 'cld')
    params, parts, kwargs = _parse_kwargs(skip_keys, skip_values, **kwargs)
    results = []
    for values in itertools.product(*params.values()):
        # Create dataset and calculate climate feedbacks
        # NOTE: Retain flux trends since they do not affect least squares regression
        # estimate, only affect uncertainty. See jupyter notebook for examples.
        kw = dict(zip(params, values))
        style = 'annual' if kw['annual'] else 'monthly'
        remove = 'average' if kw['anomaly'] else 'climate'
        source = LONGS_SOURCE[kw['source']]
        kw_filter = {**kwargs, 'anomaly': kw.pop('anomaly', True)}
        kw_fluxes = dict(style=style, components=('', 'cs'), boundaries=('t',))
        dataset = annual_filter(dataset, **kw_filter)
        result = dataset.rename({f'ts_{source[:3]}': 'ts'})
        result['ts'] = detrend_dims(result.ts)  # detrend temperature (see above)
        for values in itertools.product(*parts.values()):
            items = dict(zip(parts, values))
            cld, wav, sky = items['cld'], items['wav'], items['sky']
            cld, sky = (sky, '') if sky == 'cld' else (cld, sky)
            name = f'{cld}_r{wav}nt{sky}'.strip('_')
            result[name] = get_result(result, name, time=None)
        names = [name for name in result.data_vars if name[:3] == 'ts_']
        names += ['rlut', 'rlutcs', 'rsut', 'rsutcs', 'rsdt']
        names += ['x', 'y', 'fit', 'fit_lower', 'fit_upper']
        result = result.drop_vars(names)  # e.g. other source or 'lower' and 'upper'

        # Save feedback data
        # Facets: ('CMIP6', 'CERES', 'historical', 'r1i1p1f1')
        # Version: ('gis|had', 'monthly|annual', 'region', year1, year2)
        if output is False:
            continue
        start = dataset.time[0].dt.year.item()
        stop = dataset.time[-1].dt.year.item()
        head = 'feedbacks_Amon_CERES_historical_flagship'
        tail = f'{start:04d}-{stop:04d}-{source}-{style}-{remove}.nc'
        base = Path('~/data/ceres-feedbacks').expanduser()
        file = '_'.join((head, tail))
        if isinstance(output, Path) or '/' in (output or ''):  # copy to general.py
            base = Path(output).expanduser()
        elif output is not None:  # copy to general.py
            file = output
        if not base.is_dir():
            os.mkdir(base)
        if not base.is_dir():
            raise ValueError(f'Invalid output location {base}.')
        result = _feedbacks_from_fluxes(result, **kw_fluxes)
        print(f'Saving file: {file}')
        result.to_netcdf(base / file)
        results.append(result)
    return results


def process_scalar(dataset=None, output=None, **kwargs):
    """
    Save global climate feedbacks and uncertainty using `observed.calc_feedbacks`.

    Parameters
    ----------
    dataset : xarray.Dataset, optional
        The input dataset. Default is ``load_dataset(globe=False)``.
    output : bool or path-like, optional
        The output directory or name. If ``False`` nothing is saved.
    **kwargs
        Passed to `calc_feedbacks`.
    """
    # Create dataset and calculate feedbacks
    # NOTE: Unlike 'coupled' feedback calculations this has coordinates for several
    # variant calculations. In future should allow this for non-scalar estimates.
    if dataset is None:  # load global average observations
        dataset = load_dataset(globe=True)
    if not isinstance(dataset, xr.Dataset):
        raise ValueError('Input argument must be a dataset.')
    if output is False:  # begin printing
        print('(', end=' ')
    else:  # print message
        print('Calculating global climate feedbacks.')
    testing = kwargs.get('testing', False)
    params, parts, kwargs = _parse_kwargs(**kwargs)
    translate = kwargs.pop('translate', None)
    results = {}
    for values in itertools.product(*params.values()):
        kwarg = dict(zip(params, values))
        coord = _parse_coords(dataset.time, translate, **kwarg)
        kwarg.update(kwargs)
        years = kwarg.get('years', None)
        source = kwarg.pop('source', None) or ''
        source = source if source[:3] in ('had', 'gis') else ''
        sample = years is not None and np.isscalar(years)
        level, _ = TRANSLATE_PARAMS['internal', None]
        levels = (*coord, level, 'correct')  # see below
        result = [{}, {}] if sample else [{}]
        internals = (False, True) if sample else (False,)
        if output is False:  # succinct information
            print('-'.join(filter(None, coord.values())), end=' ')
        else:  # detailed information
            print(' '.join(f'{key} {value!r}' for key, value in coord.items()))
        for values in itertools.product(*parts.values()):
            kw = dict(nofit=True, pctile=False, **kwarg)  # pctile=False -> +/- sigma
            opts = dict(zip(parts, values))
            if not source:  # _parse_kwargs only keeps if 'names' not passed
                name, cld, wav, sky = values[0], '', '', ''
            else:  # _parse_kwargs handles incompatibilities
                name, cld, wav, sky = '', *(opts[key] for key in ('cld', 'wav', 'sky'))
            if sky == 'cld':  # workaround
                cld, sky = sky, ''
            temp = f'ts_{source[:3]}'.strip('_')
            flux = name or f'{cld}_r{wav}nt{sky}'.strip('_')
            lam, sigma1, sigma2, dof, *_ = calc_feedback(dataset, temp, flux, **kw)
            mean = lam.mean('sample') if sample else lam
            sigma = 0.5 * (sigma2 - sigma1)
            for internal in internals:
                if internal:  # spread of best-estimate across sample periods
                    error = lam.std('sample', ddof=1)  # internal sigma
                    count = xr.full_like(error, lam.sizes['sample'] - 1)
                else:  # pctile=False -> +/- sigma
                    error = sigma.mean('sample') if sample else sigma
                    count = dof.mean('sample') if sample else dof
                delta1, delta2 = var._dist_bounds(error, dof=count, pctile=True)  # 95%
                delta = xr.DataArray(delta2 - delta1, dims=error.dims)
                concat = xr.DataArray(['slope', 'sigma', 'range', 'dof'], dims='statistic')  # noqa: E501
                concat = xr.concat((mean, error, delta, count), dim=concat)
                concat.attrs.update(lam.attrs)  # update attributes
                result[internal][f'{flux}_lam'] = concat
        for idx, concat in enumerate(result):
            _, value = TRANSLATE_PARAMS['internal', bool(idx)]
            value = (*coord.values(), value)
            concat = xr.Dataset(concat)
            results[value] = concat

    # Concatenate and save data
    # NOTE: Here calc_feedack() returns array with 'correct' coordinate for speed, then
    # combine ('source', 'period', 'initial', 'style', 'remove', 'detrend', 'correct').
    if not results:
        raise RuntimeError('No datasets created.')
    result = xr.concat(
        results.values(),
        dim='concat',
        coords='minimal',
        compat='override',
        combine_attrs='override',
    )
    correct = np.atleast_1d(kwargs.get('correct', True))
    correct = xr.DataArray(correct, dims='correct')  # in case scalar
    result = result if 'correct' in result.dims else result.expand_dims('correct')
    result = result.stack(version=['concat', 'correct'])
    result = result.transpose('version', ...)
    version = tuple(results)  # original version values
    version = [(*version[num], correct) for num, correct in result.version.values]
    coord = xr.DataArray(
        pd.MultiIndex.from_tuples(version, names=levels),
        dims='version',
        name='version',
        attrs={'long_name': 'feedback version'},
    )
    result = result.assign_coords(version=coord)
    result = result.transpose('version', 'statistic', ...)
    if output is False:  # end printing
        print(')', end=' ')
    else:  # save result
        base = Path('~/data/global-feedbacks').expanduser()
        file = 'tmp.nc' if testing else 'feedbacks_CERES_global.nc'
        if isinstance(output, str) and '/' not in output:
            output = base / output
        elif output:
            output = Path(output).expanduser()
        if not output:
            output = base / file
        elif not output.suffix:
            output = output / file
        if not output.parent.is_dir():
            os.mkdir(output.parent)
        if output.is_file():
            os.remove(output)
        print(f'Saving file: {output.name}')
        result.reset_index('version').to_netcdf(output)
    return result
