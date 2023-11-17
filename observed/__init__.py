#!/usr/bin/env python3
"""
Shared utilities for working with observational data.
"""
# Internal stuff
from functools import partial
from warnings import warn
from icecream import ic, colorize
ic.configureOutput(outputFunction=lambda *args: print(colorize(*args)))
_ObservedWarning = type('ObservedWarning', (UserWarning,), {})
_warn_observed = partial(warn, category=_ObservedWarning, stacklevel=2)

# Import tools
from climopy import ureg, vreg, const  # noqa: F401
from .oscillations import *  # noqa: F401, F403
from .reanalysis import *  # noqa: F401, F403
from .station import *  # noqa: F401, F403
from .surface import *  # noqa: F401, F403
from .satellite import *  # noqa: F401, F403
