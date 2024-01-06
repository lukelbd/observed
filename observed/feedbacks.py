#!/usr/bin/env python3
"""
Get observational forcing-feedback estimates.
"""
import itertools
import os
from pathlib import Path
from icecream import ic  # noqa: F401

import numpy as np  # noqa: F401
import xarray as xr
from climopy import var, ureg, vreg  # noqa: F401
from coupled.process import get_data
from cmip_data.feedbacks import _feedbacks_from_fluxes

from .datasets import load_dataset
from .arrays import annual_average, annual_filter, detrend_series, regress_series

__all__ = [
    'annual_average',
    'annual_filter',
    'calc_feedback',
    'process_feedbacks',
    'process_scalar',
]

# Flux labels
LABELS_SRC = {
    'he': 'He et al.',
    'gis': 'GISTEMP4',
    'had': 'HadCRUT5',
}
LABELS_WAV = {
    'f': 'net',
    'l': 'longwave',
    's': 'shortwave',
    'u': 'reflected',
    'd': 'insolation',
}
LABELS_CLD = {
    '': 'all-sky',
    'cs': 'clear-sky',
    'ce': 'cloud effect',
    'cld': 'He et al. cloud',
    'cldhi': 'He et al. high-cloud',
    'cldlo': 'He et al. low-cloud',
    'cldmx': 'He et al. mixed-cloud',
}

# Keyword argument for scalar feedback calculations
OPTIONS_DEFAULT = {
    'wav': ('f', 'l', 's'),  # separate variable
    'sky': ('', 'ce', 'cld'),  # separate variable
    'src': ('gis', 'had'),
    'annual': (False, True),
    'anomaly': (False, True),
    'detrend': (3, 2, 1, 0),
    'adjust': ('r', 'y', 'x', ''),
}
OPTIONS_TESTING = {key: value[:2] for key, value in OPTIONS_DEFAULT.items()}


def get_parts(src=None, wav=None, sky=None, cld=None, sfc=None):
    """
    Return the variables and labels for the input feedback specification.

    Parameters
    ----------
    src : str or sequence, default: 'had'
        The temperature data source. One of ``'had'``, ``'gis'``, or ``'he'``.
    wav : str or sequence, default: ('f', 's', 'l')
        The flux wavelength. One of ``'f'``, ``'s'``, ``'l'``.
    sky : str, default: ''
        The flux sky component. One of ``''``, ``'cs'``, ``'ce'``.
    cld : bool or str, optional
        Whether to use masking-adjusted cloud feedbacks from He et al. paper.
    sfc : str, default: 't'
        The flux boundary. One of ``'t'``, ``'s'``, ``'a'``.
    """
    # NOTE: This is used by calc_feedback(), plot_series(), and plot_feedback(). Also
    # supports specifying empty tuples e.g. to omit plotting a series.
    sky = '' if sky is None else sky
    sfc = 't' if sfc is None else sfc
    cld = 'cld' if cld is True else cld
    srcs = ('gis', 'had') if src is None else src  # pass False or () to skip
    srcs = () if not srcs else (src,) if isinstance(src, str) else srcs
    wavs = ('f', 's', 'l') if wav is None else wav  # pass False or () to skip
    wavs = () if not wavs else (wav,) if isinstance(wav, str) else wavs
    rads = tuple(f'rs{wav}{sfc}' if wav in 'ud' else f'r{wav}n{sfc}' for wav in wavs)
    temps = tuple(f'ts_{src}' for src in srcs)
    fluxes = tuple(f'{cld}_{rad}' if cld else f'{rad}{sky}' for rad in rads)
    prefix = 'ts-' + '-'.join(srcs)
    suffix = cld if cld is not None else sky
    templabels = [LABELS_SRC[src] for src in srcs]
    fluxlabels = [f'{LABELS_WAV[wav].title()} {LABELS_CLD[suffix]}' for wav in wavs]
    pathlabel = '_'.join(filter(None, (prefix, '-'.join(rads), suffix)))
    return temps, fluxes, templabels, fluxlabels, pathlabel


def calc_feedback(
    dataset, *args, src=None, wav=None, sky=None, cld=None, sfc=None,
    annual=False, variability=False, detrend=False, verbose=False, **kwargs,
):
    """
    Return climate feedback calculation for global time series.

    Parameters
    ----------
    dataset : xarray.Dataset
        The flux and temperature data from `open_dataset`.
    *args : str, optional
        The optional explicit variable names.
    annual : bool, optional
        Whether to use annual averages.
    variability : bool or int, optional
        Whether to estimate uncertainty from internal variability or the years to use.
    detrend : bool or str, default: False
        Whether to detrend the data. Use ``'x'`` for temperature and ``'y'`` for flux.
    verbose : bool or int, optional
        Whether to print a message after detrending. ``1`` is ``R``, ``2`` is ``T``.
    src, wav, sky, cld, sfc : optional
        Passed to `get_parts`.
    **kw_annual
        Passed to `annual_average` or `annual_filter`.
    **kw_regress
        Passed to the `regress_series`.

    Returns
    -------
    lam, lam_lower, lam_upper : xarray.DataArray
        The feedbacks for the input settings.
    fit, fit_lower, fit_upper : xarray.DataArray
        The Gregory regresson fit for the input settings.
    """
    # Get the data arrays
    # NOTE: Critical to pass time=None or else time average taken automatically. Note
    # flux components e.g. 'rfnt' will be auto-derived from e.g. 'rsdt' 'rsut' 'rlut'.
    src = src or 'gis'  # used in name
    wav = wav or 'f'  # default full
    kw_names = dict(src=src, wav=wav, sky=sky, cld=cld, sfc=sfc)
    try:
        from coupled.specs import _pop_kwargs  # TODO: move this
        kw_annual = _pop_kwargs(kwargs, annual_filter)
    except ImportError:
        keys_annual = ('month', 'anomaly')
        kw_annual = {key: kwargs.pop(key) for key in keys_annual if key in kwargs}
    kw_regress = kwargs.copy()
    kw_detrend = {**kwargs, 'adjust': False}
    if len(args) not in (0, 2):
        raise ValueError(f'Expected 0 or 2 positional arguments but got {len(args)}.')
    if args:  # manual specification
        temp, flux = args
    else:
        (temp,), (flux,), *_ = get_parts(**kw_names)
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
    temp = get_data(dataset, temp, time=None)  # forget time coordinates
    flux = get_data(dataset, flux, time=None)

    # Get the feedbacks and combine results
    # NOTE: Previously embedded annual stuff inside regress_series so **kwargs would
    # do it twice but with bootstrapping need to ensure correct starting months are
    # selected or e.g. might select 19-year sample instead of 20-year sample.
    mask = ~flux.isnull() & ~temp.isnull()
    flux, temp = flux[mask], temp[mask]
    func = annual_average if annual else annual_filter
    flux, temp = func(flux, **kw_annual), func(temp, **kw_annual)
    coords = None  # fit coordinates
    starts = [0]  # regression index
    scale = 1 if annual else 12
    size = flux.size  # regression size
    lams, lams_lower, lams_upper = [], [], []
    dofs, fits, fits_lower, fits_upper = [], [], [], []
    if variability:
        size = 20 if variability is True else variability
        coords = np.linspace(np.min(temp), np.max(temp), 100)
        if size % 2:
            raise ValueError(f'Number of variability years {variability} must be even.')
        size, step = size * scale, size * scale // 2
        starts = np.arange(0, flux.size - size + 1, step)  # uses full record
    for start in starts:
        datas = []
        for idx, data in enumerate((flux, temp)):
            data = data[start:start + size]
            if idx in idxs:  # note detrend
                verb = True if verbose and start == starts[0] else False
                data = detrend_series(data, verbose=verb, **kw_detrend)
            datas.append(data)
        result = regress_series(*datas, coords=coords, **kw_regress)
        lam, lam_lower, lam_upper, dof, fit, fit_lower, fit_upper = result
        lams.append(lam)
        lams_lower.append(lam_lower)
        lams_upper.append(lam_upper)
        dofs.append(dof)
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
    # lam, dof = lams.mean(), np.mean(dofs)
    # lam_sigma = lams.std(dim='sample', ddof=1)  # unbiased normal sigma estimator
    # lam_lower, lam_upper = var._dist_bounds(lam_sigma, pctile, dof=)
    # lam_lower = xr.DataArray(lam_lower, attrs=lam.attrs)
    # lam_upper = xr.DataArray(lam_upper, attrs=lam.attrs)
    if variability:
        kw_concat = dict(dim='sample', coords='minimal', compat='override')
        coords = dict(fits[0].coords)  # standardized dependent coordinates
        lam = xr.concat(lams, **kw_concat)
        lam_lower = xr.concat(lams_lower, **kw_concat)
        lam_upper = xr.concat(lams_upper, **kw_concat)
        dof = xr.DataArray(dofs, dims='sample')
        fit = xr.concat(fits, **kw_concat).assign_coords(coords)
        fit_lower = xr.concat(fits_lower, **kw_concat).assign_coords(coords)
        fit_upper = xr.concat(fits_upper, **kw_concat).assign_coords(coords)
    lam_name = f'{flux.name}_lam_{src}'
    fit_name = f'{flux.name}_fit_{src}'
    lam.name = lam_name
    lam_lower.name = f'{lam_name}_lower'
    lam_upper.name = f'{lam_name}_upper'
    fit.name = fit_name
    fit_lower.name = f'{fit_name}_lower'
    fit_upper.name = f'{fit_name}_lower'
    return lam, lam_lower, lam_upper, dof, fit, fit_lower, fit_upper


def process_feedbacks(dataset=None, output=None, source=None, style=None, **kwargs):
    """
    Save climate feedbacks using `cmip_data.process` and `cmip_data.feedbacks`.

    Parameters
    ----------
    dataset : xarray.Dataset, optional
        The input dataset. Default is ``load_dataset(globe=True)``.
    output : path-like, optional
        The output directory.
    source : {'gistemp', 'hadcrut'}, optional
        The temperature source.
    style : {'monthly', 'annual'}, optional
        The feedback style.
    **kwargs
        Passed to `annual_filter`.
    """
    # Create dataset and calculate climate feedbacks
    # NOTE: Retain flux trends since they do not affect least squares regression
    # estimate, only affect uncertainty. See jupyter notebook for examples.
    # NOTE: Subtract regional temperature trends so that we still try to capture
    # 'internal' unperturbed signal in local feedbacks but getting global-average
    # normalized feedbacks are detrended by simple global-average trend. See
    # jupyter notebook for comparison of detrending before vs. after averging.
    style = style or 'monthly'
    source = source or 'gistemp'
    kw_feedbacks = dict(style=style, components=('', 'cs'), boundaries=('t',))
    if dataset is None:  # load local observations
        dataset = load_dataset(globe=False)
    if not isinstance(dataset, xr.Dataset):
        raise ValueError('Input argument must be a dataset.')
    if source not in ('gistemp', 'hadcrut'):
        raise ValueError("Input source must be 'gistemp' or 'hadcrut'.")
    dataset = dataset.rename({f'ts_{source[:3]}': 'ts'})
    dataset = annual_filter(dataset, **kwargs)
    for name in ('rlnt', 'rsnt', 'rlntcs', 'rsntcs'):
        data = get_data(dataset, name, time=None)
        dataset[name] = data
    drop = [key for key in dataset.data_vars if key[:2] == 'ts' and key != 'ts']
    drop += ['rlut', 'rlutcs', 'rsut', 'rsutcs', 'rsdt']
    drop += ['x', 'y', 'fit', 'fit_lower', 'fit_upper']
    dataset['ts'] = detrend_series(dataset.ts)  # detrend temperature (see above)
    dataset = dataset.drop_vars(drop)  # e.g. other source or '_lower' and '_upper'
    feedbacks = _feedbacks_from_fluxes(dataset, **kw_feedbacks)

    # Save feedback data
    # TODO: Copy below style of specifying either output folder or file to plotting.py
    # Facets: ('CMIP6', 'CERES', 'historical', 'r1i1p1f1')
    # Version: ('gis|had', 'monthly|annual', 'region', year1, year2)
    start = dataset.time[0].dt.year.item()
    stop = dataset.time[-1].dt.year.item()
    head = 'feedbacks_Amon_CERES_historical_r1i1p1f1'
    tail = f'{start:04d}-{stop:04d}-{source}-{style}.nc'
    base = Path('~/data/ceres-feedbacks').expanduser()
    file = '_'.join((head, tail))
    if isinstance(output, Path) or '/' in (output or ''):
        base = Path(output).expanduser()
    elif output is not None:
        file = output
    if not base.is_dir():
        os.mkdir(base)
    if not base.is_dir():
        raise ValueError(f'Invalid output location {base}.')
    print(f'Saving file: {file}')
    feedbacks.to_netcdf(base / file)
    return feedbacks


def process_scalar(dataset=None, output=None, testing=None, **kwargs):
    """
    Save scalar climate feedbacks and uncertainty using `observed.calc_feedbacks`.

    Parameters
    ----------
    dataset : xarray.Dataset, optional
        The input dataset. Default is ``load_dataset(globe=False)``.
    output : path-like, optional
        The output directory.
    default : bool, optional
    **kwargs
        Passed to `calc_feedbacks`.
    """
    # Create dataset and calculate feedbacks
    # NOTE: Unlike 'coupled' feedback calculations this has coordinates for several
    # variant calculations. In future should allow this for non-scalar estimates.
    if dataset is None:  # load global average observations
        dataset = load_dataset(globe=True)
    kwargs = {
        key: value if isinstance(value, (list, tuple)) else (value,)
        for key, value in kwargs.items()
    }
    options = OPTIONS_TESTING if testing else OPTIONS_DEFAULT
    kwargs = kwargs or options
    names, feedbacks = [], []
    print('Calculating feedbacks:', end=' ')
    for values in itertools.product(*kwargs.values()):
        # print('Options:', ', '.join(f'{key}={val!r}' for key, val in kwargs.items()))
        kw = dict(zip(kwargs, values))
        kw.update(verbose=False)
        kw['cld'] = kw.pop('sky') if 'cld' in kw.get('sky') else ''
        lam, spread1, spread2, dof, *_ = calc_feedback(dataset, pctile=True, **kw)
        lam, sigma1, sigma2, *_ = calc_feedback(dataset, pctile=False, **kw)
        cld, wav, sky = kw.pop('cld', ''), kw.pop('wav', 'f'), kw.pop('sky', '')
        name = f'{cld}_r{wav}nt{sky}'.strip('_')
        if name not in names:
            print(name, end=' ')
            names.append(name)
        kw['source'] = kw.pop('src', 'gis')
        kw['style'] = kw.pop('annual', '') and 'annual' or 'monthly'
        coords = {
            key: xr.DataArray([value], name=key, dims=key).astype(object)
            for key, value in kw.items() if key != 'verbose'
        }
        sigma = 0.5 * (sigma2 - sigma1)
        spread = 0.5 * (spread2 - spread1)
        feedback = xr.Dataset(coords=coords)
        feedback = feedback.astype(object)  # everything object
        feedback[f'{name}_lam'] = lam.expand_dims(tuple(coords))
        feedback[f'{name}_dof'] = dof.expand_dims(tuple(coords))
        feedback[f'{name}_sigma'] = sigma.expand_dims(tuple(coords))
        feedback[f'{name}_spread'] = spread.expand_dims(tuple(coords))
        feedbacks.append(feedback)
    feedbacks = xr.combine_by_coords(feedbacks)

    # Save feedback data
    # TODO: Copy below style of specifying either output folder or file to plotting.py
    # Version: ('gis|had', 'monthly|annual', 'region', year1, year2)
    # Facets: ('CMIP6', 'CERES', 'historical', 'r1i1p1f1')
    print()
    start = dataset.time[0].dt.year.item()
    stop = dataset.time[-1].dt.year.item()
    head = 'feedbacks_Amon_CERES_historical_r1i1p1f1'
    tail = f'{start:04d}-{stop:04d}-global.nc'
    base = Path('~/data/ceres-feedbacks').expanduser()
    file = '_'.join((head, tail))
    if isinstance(output, Path) or '/' in (output or ''):
        base = Path(output).expanduser()
    elif output is not None:
        file = output
    if not base.is_dir():
        os.mkdir(base)
    if not base.is_dir():
        raise ValueError(f'Invalid output location {base}.')
    print(f'Saving file: {file}')
    ic(feedbacks)
    feedbacks.to_netcdf(base / file)
    return feedbacks
