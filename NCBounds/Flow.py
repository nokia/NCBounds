#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This file is part of the pybgl project.
# https://github.com/nokia/NCBounds

__author__     = "Anne Bouillard"
__maintainer__ = "Anne Bouillard"
__email__      = "anne.bouillard@nokia-bell-labs.com"
__copyright__  = "Copyright (C) 2019, Nokia"
__license__    = "BSD-3"

from NCBounds.ArrivalCurve import ArrivalCurve
from typing import List, Tuple


class Flow:
    """
    Constructor for a flow. A flow is defined by

    :param acurve: the arrival curve for the flow
    :type acurve: ArrivalCurve
    :param path: the path it follows, that is the list of servers it crosses, represented by their number
    :type path: List[int]

    >>> acurve = ArrivalCurve(4,3)
    >>> path = [0,1,3]
    >>> Flow(acurve, path)
    <Flow: α(t) = 4.00 + 3.00 t; π = [0, 1, 3]>
    <BLANKLINE>
    >>> Flow(acurve, path).length
    3
    """
    def __init__(self, acurve=ArrivalCurve(), path=list()):
        self.acurve = acurve
        self.path = path
        self.length = len(path)

    def __str__(self) -> str:
        return "α(t) = %s; π = %s" % (self.acurve, self.path)

    def __repr__(self) -> str:
        return "<Flow: %s>\n" % self.__str__()
