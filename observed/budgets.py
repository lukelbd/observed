#!/usr/bin/env python3
"""
Functions for loading carbon station data.
"""
from pathlib import Path

import pandas as pd
import pint_pandas  # noqa: F401  # define accessor
from climopy import const, ureg, vreg  # noqa: F401
from icecream import ic  # noqa: F401

__all__ = ['load_budget', 'load_budgets']

# Assign climopy unit registry
# See: https://github.com/hgrecco/pint-pandas/issues/136
# NOTE: Using data['column'] fails with dataframes quantified outside of pint-pandas
# interface (Series are not supported). Instead use data[['column']] to maintain the
# dataframe or data.column or data.get('column') to bypass the overridden Quantity
# __getitem__ indexer and automatically strip the units before returning the series.
# NOTE: Pint-pandas is rough right now. Does not natively support other registries, can
# only assign units by calling .quantify() on data with multi-index units or via strange
# .astype('pint[units]') syntax, and wraps all data inside pint_array.PintArray objects
# that pretty printers don't recognize (so unit information is not shown) and triggers
# unit stripped warnings. Instead use ureg.Quantity(dataframe, units) syntax.
pint_pandas.PintType.ureg = ureg

# Spreadsheet parsing
SHEET_PAGES = {
    'recent': 1,
    'historical': 2,
    'ocean': 5,
    'land': 6,
    'fossil': 3,
    'land-use': 4,
    'cement': 7,
}
SHEET_NAMES = {page: name for name, page in SHEET_PAGES.items()}

# Spreadsheet properties
SHEET_HEADERS = {
    'recent': 21,
    'historical': 15,
    'ocean': 30,
    'land': 27,
    'fossil': 8,
    'land-use': 36,
    'cement': 9,
}
SHEET_LABELS = {
    'recent': 'Global Carbon Budget',
    'historical': 'Historical Budget',
    'fossil': 'Fossil Emissions by Category',
    'land-use': 'Land-Use Change Emissions',
    'ocean': 'Ocean Sink',
    'land': 'Terrestrial Sink',
    'cement': 'Cement Carbonation Sink',
}


def _parse_columns(columns):
    """
    Parse and translate the carbon budget spreadsheet columns.

    Parameters
    ----------
    columns : array-like of tuples
        The input columns as an index or array.

    Returns
    -------
    index : pandas.Index
        The parsed column index.
    units : pint.Unit
        The units for the data.
    """
    # NOTE: Sheets based on models will return multi-index results. Access budget
    # estimate published in paper using data['budget'] or data['budget', 'mean'].
    index = []
    columns = getattr(columns, 'values', columns)
    multi = any('models' in name0.lower() for name0, *_ in columns)
    units = columns[0][0]  # unit string is at top left (see above)
    for names in columns:
        if len(names) == 3:
            name0, name1, name2 = names
        elif len(names) == 2:
            name0, name1, name2 = '', *names
        else:
            raise ValueError(f'Unexpected level count {len(names)}.')
        if 'models' in name0.lower():  # land-use model entries have empty name2
            name1, name2 = name0, name1
        if 'models' in name1.lower():  # land-use, ocean sink, land sink 'models'
            name1 = 'models'
        if 'products' in name1.lower():  # ocean sink 'data-products'
            name1 = 'products'
        if name1 in ('models', 'products'):
            if any(s in name2.lower() for s in ('spread', 'sd ')):  # sd from 'products'
                name2 = f'{name1[:-1]} sigma'
            if any(s in name2.lower() for s in ('mean', 'average')):  # avg 'products'
                name2 = f'{name1[:-1]} mean'
        else:
            for part, replace in (  # simple translations
                ('atmospheric', 'atmos'),
                ('capita', 'per_capita'),
                ('land_use', 'land-use'),
                ('fossil.', 'total'),  # standardize
                ('sigma', 'sigma'),  # standardize
                ('net', 'mean'),  # standardize
            ):
                if part in name2.lower():
                    name2 = replace
            if name1 == units or 'GCB' in name1:  # any spreadsheet
                name1 = 'budget'
            if 'GCB' in name2:  # ocean sink, land sink
                name2 = 'mean'
            if 'gross' in name2.lower():  # land use parts
                name2 = name2.lower().split()[-1]
            if not any(s == name2 for s in ('Cao', 'Guo')):
                name2 = name2.lower()
            for sep in (' ', '.'):  # net budget, fossil fuel, etc.
                name2 = name2.split(sep)[0]
        if multi:
            index.append((name1, name2))
        else:
            index.append(name2)
    index = pd.Index(index)  # tuple inputs result in multi-index
    units = ureg.parse_units(units)
    return index, units


def load_budgets(*sheets, **kwargs):
    """
    Load and combine annual carbon budget spreadsheets.

    Parameters
    ----------
    *sheets : str or int, optional
        The sheet name(s) or number(s). Defaults to all.
    **kwargs
        Passed to `load_budget`.

    Returns
    -------
    data : pandas.DataFrame
        The combined data.
    """
    # NOTE: This is used by model.py _budget_forcing() to create forcing datasets
    # NOTE: Here 'historical' has preliminary 2023 data so prefer to other sheets
    def _merge_columns(names):
        if names[0] in ('budget', 'model', 'models', 'product', 'products'):
            columns = names[-1]
        else:  # use unmodified label for e.g. mean models and bookkeeping methods
            columns = ' '.join(key for key in names if key != 'mean')
        return columns
    sheets = sheets or list(SHEET_PAGES)
    sheets = [SHEET_NAMES.get(sheet, sheet) for sheet in sheets]
    if any(sheet not in SHEET_PAGES for sheet in sheets):
        raise ValueError(f'Invalid sheet name(s): {sheets!r}')
    if 'recent' in sheets and 'historical' in sheets:
        sheets.remove('recent')  # default to full record
    if 'recent' not in sheets and 'historical' not in sheets:
        sheets.append('recent')  # required for residuals
    datas = {}
    sheets = [sheet for sheet in SHEET_PAGES if sheet in sheets]  # standard order
    for sheet in sheets:  # first sheet is budget
        data = load_budget(sheet, **kwargs)
        if sheet in ('recent', 'historical'):
            units = data.units
            data = data.magnitude
            for name in data.columns:  # apply default dataframes
                rename = {name: 'total' if name == 'fossil' else 'mean'}
                idata = data[[name]].rename(columns=rename)
                idata = pd.concat([idata], names=[name], axis='columns')
                datas[name] = idata
        else:  # concatenate column levels
            data = data.to(units)
            data = data.magnitude
            if isinstance(data.columns, pd.MultiIndex):
                index = data.columns.map(_merge_columns)
                data.columns = index
            if sheet in datas:  # prefer recent or historical
                data = pd.merge(datas[sheet], data, how='left')
                data.index = datas[sheet].index
            datas[sheet] = data
    data = pd.concat(datas.values(), keys=datas.keys(), axis=1)
    data = ureg.Quantity(data, units)
    return data


def load_budget(sheet=None, base=None, year=None):
    """
    Load annual carbon budget spreadsheet.

    Parameters
    ----------
    name : str or int, optional
        The sheet name or number.
    base : path-like
        The directory name.
    year : int, optional
        The budget year.

    Returns
    -------
    data : pandas.DataFrame
        The budget data.
    """
    # TODO: Switch to pint-pandas support for quantifying underlying data in future
    # if pretty printer and other support improves. See above notes.
    # NOTE: Using data.columns.levels shows unique level values, but may be unsorted,
    # so use get_level_values() to retrieve unit indicator in upper left. Note land
    # use spreadsheet includes headers with extra info on same line that we drop.
    year = year or 2023
    version = 1.1 if year == 2023 else 1.0
    file = f'Global_Carbon_Budget_{year}v{version}.xlsx'
    base = Path(base or '~/data/carbon-budget').expanduser()
    path = base / file
    sheet = sheet or 1  # global budget
    page = SHEET_PAGES.get(sheet, sheet)
    if isinstance(page, str):
        options = ', '.join(map(repr, SHEET_PAGES))
        raise ValueError(f'Unknown sheet {sheet!r}. Options are: {options}.')
    name = SHEET_NAMES.get(page, page)
    label = SHEET_LABELS[name]
    start = SHEET_HEADERS[name]
    land_use = page == 4  # three headers for land-use
    header = range(start - 1, start + 1 + land_use)
    data = pd.read_excel(
        path,
        header=tuple(header),
        index_col=0,  # all have same index
        sheet_name=page,  # sheet number
    )
    data = data.dropna(axis=1, how='all')  # empty divider columns
    data.index.name = 'year'  # lower-case and assign if missing
    index, units = _parse_columns(data.columns)
    data.columns = index  # include units index level in future
    if land_use:  # correct model data sign
        mask1 = index.get_level_values(0) == 'models'
        mask2 = ~index.get_level_values(1).str.startswith('model')
        mask = mask1 & mask2  # model columns excluding mean
        data.iloc[:, mask] *= -1
    if 'atmos' in index:
        names = ['atmos']  # enforce standard order
        names.extend(name for name in SHEET_PAGES if name in index)
        names.extend(name for name in index if name not in names)
        data = data[names]  # enforce standard column order
    data = ureg.Quantity(data, units)  # data.pint.quantify() in future
    data.name = name  # internal name
    data.label = label  # plotted label
    if 'per_capita' in data.columns:  # tons carbon per capita per year
        scale = ureg.Quantity(1, 'tC / yr') / units
        scale = scale.to('dimensionless')
        data.per_capita *= scale.magnitude
    return data
