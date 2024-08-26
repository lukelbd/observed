#!/usr/bin/env python3
"""
Shared utilities for working with observational data.
"""
# Internal stuff
from functools import partial
from warnings import warn
from icecream import ic, colorize, pprint
_ObservedWarning = type('ObservedWarning', (UserWarning,), {})
_warn_observed = partial(warn, category=_ObservedWarning, stacklevel=2)
ic.configureOutput(
    outputFunction=lambda *args: print(colorize(*args)),
    argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False),
)

# Import tools
from climopy import ureg, vreg, const  # noqa: F401
from .arrays import *  # noqa: F401, F403
from .budgets import *  # noqa: F401, F403
from .datasets import *  # noqa: F401, F403
from .feedbacks import *  # noqa: F401, F403
from .indices import *  # noqa: F401, F403
from .reanalysis import *  # noqa: F401, F403
from .stations import *  # noqa: F401, F403
from .surface import *  # noqa: F401, F403
from .satellite import *  # noqa: F401, F403
from .tables import *  # noqa: F401, F403
