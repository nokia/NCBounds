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
from numpy import Inf


class ServiceCurve:
    """
    Constructor for rate-latency service curve. in the following, we denote :math:`\\beta: t \\mapsto R(t-T)_+`
    such a curve. An argument of a function name sc1 will be denoted :math:`\\beta_1: t \\mapsto R_1(t-T_1)_+` and
    similarly for all indices.

    :param rate: :math:`R`
    :param latency: :math:`T`

    >>> service_curve = ServiceCurve(rate =5., latency=3.)
    """
    def __init__(self, rate: float = 0, latency: float = 0):
        self.rate = rate
        self.latency = latency

    def __repr__(self):
        return '<ServiceCurve: %s>' % self

    def __str__(self):
        return "%.2f . (t - %.2f)+" % (self.rate, self.latency)


def convolution(sc1: ServiceCurve, sc2: ServiceCurve) -> ServiceCurve:
    r"""
    Computes the (min,plus)-convolution of two rate-latency service curves:

    .. math:: \beta\gets\beta_1 * \beta_2

    and :math:`\beta: t \mapsto R(t-T)_+` with

    * :math:`R = \min (R_1, R_2)`;
    * :math:`T = T_1 + T_2`;

    :param sc1: first sevice curve.
    :type sc1: ServiceCurve
    :param sc2: second service curve
    :type sc2: ServiceCurve
    :return: The convolution of the two service curves _sc1_ * _sc2_.
    :rtype: ServiceCurve

    >>> sc1 = ServiceCurve(5, 3)
    >>> sc2 = ServiceCurve(2, 2)
    >>> convolution(sc1, sc2)
    <ServiceCurve: 2.00 . (t - 5.00)+>
    """
    return ServiceCurve(rate=min(sc1.rate, sc2.rate), latency=sc1.latency + sc2.latency)


def deconvolution(ac: ArrivalCurve, sc: ServiceCurve) -> ArrivalCurve:
    r"""
    Computes the (min,plus)-convolution of an arrival curve and of a service curve:

    .. math:: \alpha'\gets\beta \oslash \alpha

    and :math:`\alpha': t \mapsto \sigma' + \rho' t` with

    * :math:`\sigma' = \sigma + \rho T`;
    * :math:`\rho' = \rho`;

    if :math:`R \geq \rho` and :math:`\alpha': t \mapsto \infty` otherwise.

    :param ac: Arrival curve of the flow.
    :type ac: ArrivalCurve
    :param sc: service curve
    :type sc: ServiceCurve of the server
    :return: The deconvolution of _ac_ by _sc_, that is an arrival curve for the output flow
    :rtype: ArrivalCurve

    >>> ac = ArrivalCurve(2, 2)
    >>> sc = ServiceCurve(5, 3)
    >>> deconvolution(ac, sc)
    <ArrivalCurve: 8.00 + 2.00 t>

    >>> ac = ArrivalCurve(5, 3)
    >>> sc = ServiceCurve(2, 2)
    >>> deconvolution(ac, sc)
    <ArrivalCurve: inf + inf t>
    """
    if ac.rho > sc.rate:
        return ArrivalCurve(sigma=Inf, rho=Inf)
    else:
        return ArrivalCurve(sigma=ac.sigma + ac.rho * sc.latency, rho=ac.rho)


def delay(ac: ArrivalCurve, sc: ServiceCurve) -> float:
    r"""
    Computes the maxumum delay bound for a flow with arrival curce ac traversing a server with service curve sc :

    .. math:: D \gets T + \frac{\sigma}{R}

    if :math:`R \geq \rho` and :math:`D = +\infty` otherwise.

    :param ac: Arrival curve.
    :type ac: ArrivalCurve
    :param sc: service curve
    :type sc: ServiceCurve
    :return: A maximum delay bound of the flow.
    :rtype: float

    >>> ac = ArrivalCurve(2, 2)
    >>> sc = ServiceCurve(5, 3)
    >>> delay(ac, sc)
    3.4

    >>> ac = ArrivalCurve(5, 3)
    >>> sc = ServiceCurve(2, 2)
    >>> delay(ac, sc)
    inf
    """

    if ac.rho > sc.rate:
        return Inf
    else:
        return (ac.sigma + sc.rate * sc.latency) / sc.rate


def backlog(ac: ArrivalCurve, sc: ServiceCurve) -> float:
    r"""
    Computes the maxumum backlog bound for a flow with arrival curce ac traversing a server with service curve sc:

    .. math:: B \gets \sigma + \rho T

    if :math:`R \geq \rho` and :math:`B = +\infty` otherwise.

    :param ac: Arrival curve.
    :type ac: ArrivalCurve
    :param sc: service curve
    :type sc: ServiceCurve
    :return: A maximum backlog bound of the flow.
    :rtype: float

    >>> ac = ArrivalCurve(2., 2.)
    >>> sc = ServiceCurve(5., 3.)
    >>> backlog(ac, sc)
    8.0

    >>> ac = ArrivalCurve(5, 3)
    >>> sc = ServiceCurve(2, 2)
    >>> backlog(ac, sc)
    inf
    """
    if ac.rho > sc.rate:
        return Inf
    else:
        return ac.sigma + ac.rho * sc.latency


def residual_blind(ac: ArrivalCurve, sc: ServiceCurve) -> ServiceCurve:
    r"""
    Computes an arrival curve for the output flow that corresponds to a flow with arrival curce ac traversing a
    server with service curve sc.

    :math:`\beta_r \gets = R_r(t-T_r)_+`

    where

    * :math:`R_r = R-\rho` and
    * :math:`T_r = \frac{\sigma + RT}{R-\rho}`

    if :math:`R \geq \rho` and :math:`B = +\infty` otherwise.

    :param ac: Arrival curve.
    :type ac: ArrivalCurve
    :param sc: service curve
    :type sc: ServiceCurve
    :return: the residual service curve.
    :rtype: ServiceCurve

    >>> ac = ArrivalCurve(2., 2.)
    >>> sc = ServiceCurve(5., 3.)
    >>> residual_blind(ac, sc)
    <ServiceCurve: 3.00 . (t - 5.67)+>

    >>> ac = ArrivalCurve(5, 3)
    >>> sc = ServiceCurve(2, 2)
    >>> residual_blind(ac, sc)
    <ServiceCurve: 0.00 . (t - 0.00)+>
    """
    if ac.rho >= sc.rate:
        return ServiceCurve(rate=0., latency=0.)
    else:
        return ServiceCurve(
            rate=sc.rate - ac.rho,
            latency=(ac.sigma + sc.rate * sc.latency) / (sc.rate - ac.rho)
        )
