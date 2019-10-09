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


from NCBounds.ServiceCurve import ServiceCurve, deconvolution, residual_blind
from NCBounds.ArrivalCurve import ArrivalCurve, ac_sub, ac_sum


class Server:
    """
    Constructor for a server. A server is defined by a service curve and a service policy. By default, the service
    policy is blind, and servers with other service policies are defined as subclasses

    :param scurve: the service curve of the server
    :type scurve: ServiceCurve

    >>> scurve = ServiceCurve(5.,4.)
    >>> Server(scurve)
    <Server: β(t) = 5.00 . (t - 4.00)+>
    <BLANKLINE>
    """
    def __init__(self, scurve=ServiceCurve()):
        self.scurve = scurve

    def __str__(self) -> str:
        return "β(t) = %s" % self.scurve

    def __repr__(self) -> str:
        return "<Server: %s>\n" % self.__str__()

    def list_residual_output(self, list_ac: list) -> tuple:
        """
        Computes the residual service curve and arrival curve of the departure process for each flow
        of the server is crossed by flows described by their arrival curves in list_ac

        :param list_ac: list of arrival curves
        :type list_ac: list[ArrivalCurve]
        :return: the list of residual service curves  and the list of output arrival curves for each departure
            process of a flow
        :rtype: (list(ServiceCurve), list(ArrivalCurve)

        >>> scurve = ServiceCurve(5., 4.)
        >>> server = Server(scurve)
        >>> list_ac = [ArrivalCurve(2., 1.), ArrivalCurve(3., 2.)]
        >>> server.list_residual_output(list_ac)
        ([<ServiceCurve: 3.00 . (t - 7.67)+>, <ServiceCurve: 4.00 . (t - 5.50)+>], [<ArrivalCurve: 9.67 + 1.00 t>, \
<ArrivalCurve: 14.00 + 2.00 t>])
        """
        list_res = []
        list_outac = []
        ac = ac_sum(list_ac)
        for i in range(len(list_ac)):
            ac_cross = ac_sub(ac, list_ac[i])
            sc_res = residual_blind(ac_cross, self.scurve)
            list_res += [sc_res]
            list_outac += [deconvolution(list_ac[i], sc_res)]
        return list_res, list_outac


# class FIFOServer(Server):
#     def list_residual_output(self, list_ac):
#         """
#         Computes the residual service curves under the FIFO policy for every flow entering the server
#         :param list_ac: list of arrival curves of the flows crossing the server
#         :type list_ac: ArrivalCurve list
#         :return: list of residual service curves for each flow and the list of output arrival curves
#         :rtype: ServiceCurve list * ArrivalCurve list
#         """
#         list_res = []
#         list_outac = []
#         # if self.policy == BLIND:
#         ac = ac_sum(list_ac)
#         # print("sum ar", ac)
#         for i in range(len(list_ac)):
#             ac_cross = ac_sub(ac, list_ac[i])
#             # print("ar cross", ac_cross)
#             sc_res = residual_blind(ac_cross, self.scurve)
#             list_res += [sc_res]
#             list_outac += [deconvolution(list_ac[i], sc_res)]
#             # print("********", list_res[i], list_outac[i])
#         return list_res, list_outac
#
#
# class PriorityServer(Server):
#     def __init__(self, scurve=ServiceCurve(), priority=list()):
#         """
#
#         :param scurve: service curve of the server
#         :type scurve: ServiceCurve
#         :param priority: priority order on the flows
#         :type priority: int list
#         """
#         self.scurve = scurve
#         self.priority = priority
#
#     def list_residual_output(self, list_ac):
#         """
#         Computes the residual service curves under the FIFO policy for every flow entering the server
#         :param list_ac: list of arrival curves of the flows crossing the server
#         :type list_ac: ArrivalCurve list
#         :return: list of residual service curves for each flow and the list of output arrival curves
#         :rtype: ServiceCurve list * ArrivalCurve list
#         """
#         list_res = []
#         list_outac = []
#         # if self.policy == BLIND:
#         ac = ac_sum(list_ac)
#         # print("sum ar", ac)
#         for i in range(len(list_ac)):
#             ac_cross = ac_sub(ac, list_ac[i])
#             # print("ar cross", ac_cross)
#             sc_res = residual_blind(ac_cross, self.scurve)
#             list_res += [sc_res]
#             list_outac += [deconvolution(list_ac[i], sc_res)]
#             # print("********", list_res[i], list_outac[i])
#         return list_res, list_outac
