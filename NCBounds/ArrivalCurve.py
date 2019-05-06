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

from numpy import Inf
from typing import List


class ArrivalCurve:
    """
    The ArrivalCurve class encodes token-bucket arrival curves.
    :math:`\\alpha: t \\mapsto \\sigma + \\rho t`.

    :param sigma: burst of the arrival curve
    :param rho: long-term arrival rate

    >>> arrival_curve = ArrivalCurve(sigma =5., rho=3.)
    """

    def __init__(self, sigma: float = Inf, rho: float = Inf):
        self.sigma = sigma
        self.rho = rho

    def __repr__(self):
        return '<ArrivalCurve: %s>' % self

    def __str__(self) -> str:
        return "%.2f + %.2f t" % (self.sigma, self.rho)


def ac_add(ac1: ArrivalCurve, ac2: ArrivalCurve) -> ArrivalCurve:
    r"""
    Makes the sum of two arrival curves:

    .. math:: \alpha\gets\alpha_1 + \alpha_2

    :param ac1: first arrival curve
    :type ac1: ArrivalCurve
    :param ac2: Second arrival curve
    :type ac2: ArrivalCurve
    :return: The sum of the two arrival curves
    :rtype: ArrivalCurve

    >>> ac1 = ArrivalCurve(5, 3)
    >>> ac2 = ArrivalCurve(2, 2)
    >>> ac_add(ac1, ac2)
    <ArrivalCurve: 7.00 + 5.00 t>
    """
    return ArrivalCurve(sigma=ac1.sigma + ac2.sigma, rho=ac1.rho + ac2.rho)


def ac_sum(list_ac: List[ArrivalCurve]) -> ArrivalCurve:
    r"""
    Makes the sum of all the arrival curves in :param list_ac:

    .. math:: \alpha \gets \sum_{i=1}^n \alpha_i

    :param list_ac: a first arrival curve
    :type list_ac: ArrivalCurve list
    :return: The sum of the arrival curves in :param list_ac:
    :rtype: ArrivalCurve

    >>> ac1 = ArrivalCurve(5, 3)
    >>> ac2 = ArrivalCurve(2, 2)
    >>> ac_sum([ac1, ac2])
    <ArrivalCurve: 7.00 + 5.00 t>
    """
    return ArrivalCurve(sigma=sum(x.sigma for x in list_ac), rho=sum(x.rho for x in list_ac))


def ac_sub(ac1: ArrivalCurve, ac2: ArrivalCurve) -> ArrivalCurve:
    r"""
    Makes the sum of two arrival curves:
    _Warning:_ should be used for removing a flow from the aggregation only

    .. math:: \alpha\gets\alpha_1 - \alpha_2

    :param ac1: first arrival curve
    :type ac1: ArrivalCurve
    :param ac2: Second arrival curve
    :type ac2: ArrivalCurve
    :return: The difference between the two arrival curves
    :rtype: ArrivalCurve

    >>> ac1 = ArrivalCurve(5, 3)
    >>> ac2 = ArrivalCurve(2, 2)
    >>> ac_sub(ac1, ac2)
    <ArrivalCurve: 3.00 + 1.00 t>
    """
    if ac2.sigma == Inf or ac2.rho == Inf:
        return ArrivalCurve(Inf, Inf)
    else:
        return ArrivalCurve(sigma=ac1.sigma - ac2.sigma, rho=ac1.rho - ac2.rho)
