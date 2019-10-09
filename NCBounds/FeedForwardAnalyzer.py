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

from NCBounds.Network import Network, topological_sort, ToyNetwork, ToyTreeNetwork
from NCBounds.ServiceCurve import ServiceCurve, convolution, delay, backlog
from NCBounds.ArrivalCurve import ArrivalCurve
import numpy as np


class FeedForwardAnalyzer:
    """
    Constructor for the feed-forward analyzer of a network. The network should be feed-forward.
    This class regroup several subclasses, depending of the typs of analyze to perform.

    * SFA: separate flow analysis, that computes a residual service curve and output arrival curve for server and
        each flow that crosses that server
    * Exact, that computes the exact bounds based on [1] and [2]

    :param network: the network to analyze
    :type network: Network
    """
    def __init__(self, network=Network()):
        self.network = network

    def delay(self, flow: int) -> float:
        """
        Computes a delay bound of a flow based on the chosen analysis.

        :param flow: flow for which the delay is computed
        :type flow: int
        :return: the delay of that flow
        :rtype: float
        """
        raise NotImplemented

    def backlog(self, flow, server):
        """
        Computes a backlog bound of a flow at a server based on the chosen analysis.

        :param flow: flow for which the backlog is computed
        :type flow: int
        :param server: server at which the backlog is computed
        :type server: int
        :return: the backlog of flow and server
        :rtype: float
        """
        raise NotImplemented


class SFAFeedForwardAnalyzer(FeedForwardAnalyzer):
    @property
    def sfa_blind(self) -> tuple:
        """
        Computes the residual service curves and the arrival curves for every flow and every server.
        :return: tab_ac is the list of output arrival curves and tab_sc: the list of residual service curves
        :rtype: tuple

        >>> toy = ToyNetwork()
        >>> toy_saf = SFAFeedForwardAnalyzer(toy)
        >>> toy_saf.sfa_blind
        ([<ArrivalCurve: 1.00 + 2.00 t>, <ArrivalCurve: 7.00 + 2.00 t>, <ArrivalCurve: 24.00 + 2.00 t>, \
<ArrivalCurve: 0.00 + 0.00 t>, <ArrivalCurve: 2.00 + 3.00 t>, <ArrivalCurve: 24.00 + 3.00 t>, \
<ArrivalCurve: 1.00 + 1.00 t>, <ArrivalCurve: 0.00 + 0.00 t>, <ArrivalCurve: 5.50 + 1.00 t>], \
[<ServiceCurve: 3.00 . (t - 3.00)+>, <ServiceCurve: 2.00 . (t - 8.50)+>, <ServiceCurve: 3.00 . (t - 12.17)+>, \
<ServiceCurve: inf . (t - 0.00)+>, <ServiceCurve: 3.00 . (t - 7.33)+>, <ServiceCurve: 4.00 . (t - 9.12)+>, \
<ServiceCurve: 2.00 . (t - 4.50)+>, <ServiceCurve: inf . (t - 0.00)+>, <ServiceCurve: 2.00 . (t - 27.50)+>])

        """
        n = self.network.num_servers
        m = self.network.num_flows
        tab_ac = (n * m) * [ArrivalCurve(0, 0)]
        tab_sc = (n * m) * [ServiceCurve(np.inf, 0)]
        list_idx_flows = np.zeros(m, int)
        sort = topological_sort(self.network.adjacency_matrix)
        for h in sort:
            for i in self.network.flows_in_servers[h]:
                if list_idx_flows[i] == 0:
                    tab_ac[i * n + h] = self.network.flows[i].acurve
            list_ac = [tab_ac[j * n + h] for j in self.network.flows_in_servers[h]]
            lr, loa = self.network.servers[h].list_residual_output(list_ac)
            for i in range(len(lr)):
                j = self.network.flows_in_servers[h][i]
                tab_sc[j * n + h] = lr[i]
                if list_idx_flows[i] < self.network.flows[i].length - 1:
                    tab_ac[j * n + self.network.flows[j].path[list_idx_flows[j] + 1]] = loa[i]
                list_idx_flows[j] += 1
        return tab_ac, tab_sc

    def delay(self, flow) -> float:
        """
        Computes a worst-case delay obound of a flow i in the network with the SFA method

        :param flow: flow under consideration
        :type flow: int
        :return: a delay upper bound for the flow
        :rtype: float

        >>> toy = ToyNetwork()
        >>> toy_saf = SFAFeedForwardAnalyzer(toy)
        >>> toy_saf.delay(0)
        24.166666666666664
        """

        sc = ServiceCurve(np.inf, 0)
        tab_ac, tab_sc = self.sfa_blind
        for h in self.network.flows[flow].path:
            print(h)
            sc = convolution(sc, tab_sc[flow * self.network.num_servers + h])
        return delay(self.network.flows[flow].acurve, sc)

    def backlog(self, flow, server):
        """
        Computes a worst-case backlog bound of a flow in a server in the network with the SFA method

        :param flow: flow under consideration
        :type flow: int
        :param server: server under consideration
        :type server: int
        :return: a backlog upper bound for the flow at the server
        :rtype: float

        >>> toy = ToyNetwork()
        >>> toy_saf = SFAFeedForwardAnalyzer(toy)
        >>> toy_saf.backlog(0, 2)
        48.33333333333333

        >>> tree = ToyTreeNetwork()
        >>> tree_sfa = SFAFeedForwardAnalyzer(tree)
        >>> tree_sfa.backlog(0, 2)
        38.2
        """
        tab_ac, tab_sc = self.sfa_blind
        return backlog(tab_ac[flow * self.network.num_servers + server],
                       tab_sc[flow * self.network.num_servers + server])


class PMOOFeedForwardAnalyzer(FeedForwardAnalyzer):
    def pmoo_blind(self, flow):
        """
        Computes the residual service curves for a flow.
        :param flow: flow under consideration
        :type flow: Flow
        :return: the residual service service curve of foi
        :rtype: ServiceCurve
        """
        n = self.network.flows[flow].path[-1]
        nk = self.network.trim(n)
        if nk.is_forest:
            res_rates = nk.residual_rate[flow]
            R = min([res_rates[j] for j in nk.flows[flow].path])
            C = sum([(nk.flows[i].acurve.sigma + nk.flows[i].acurve.rho *
                     sum([nk.servers[k].scurve.latency for k in nk.flows[i].path]))
                    for i in range(nk.num_flows) if not i == flow])
            T = sum([nk.servers[k].scurve.latency for k in nk.flows[flow].path])
            return ServiceCurve(rate=R, latency= T + C / R)
        else:
            raise NameError("Network is not a forest, PMOO analysis impossible")

    def delay(self, flow):
        return delay(self.network.flows[flow].acurve, self.pmoo_blind(flow))

    def backlog(self, flow, server):
        nk = self.network.trim(server)
        return backlog(self.network.flows[flow].acurve, PMOOFeedForwardAnalyzer(nk).pmoo_blind(flow))



class ExactFeedForwardAnalyzer(FeedForwardAnalyzer):
    def rstar(self, j, foi) -> float:
        """
        Computes the arrival rate for the flows of interests that cross server j

        :param j:  number of the server
        :type j: int
        :param foi: list of the flows of interest
        :type foi: list
        :return: the arrival rate of the flows of interest at server j
        :rtype: float

        >>> tree = ToyTreeNetwork()
        >>> tree_exact = ExactFeedForwardAnalyzer(tree)
        >>> tree_exact.rstar(1, [0])
        2
        """
        list_j = [i for i in self.network.flows_in_servers[j] if i in foi]
        return sum(self.network.flows[i].acurve.rho for i in list_j)

    def xi_rate(self, foi, j, k) -> float:
        """
        The arrival rates of the flows not of interest crossing server j and ending at server k

        :param foi: list of the flows of interest
        :type foi: list
        :param j: number of the server
        :type j: int
        :param k: ending server
        :type k: int
        :return: The sum of the rates of the flows in flows_in_server[j] \ foi ending at server k
        :rtype: float

        >>> tree = ToyTreeNetwork()
        >>> tree_exact = ExactFeedForwardAnalyzer(tree)
        >>> tree_exact.xi_rate([0], 1, 2)
        3
        """
        s = 0
        for i in self.network.flows_in_servers[j]:
            if i not in foi and self.network.flows[i].path[-1] == k:
                s += self.network.flows[i].acurve.rho
        return s

    def exact_xi(self, flows_interest, destination) -> np.ndarray:
        """
        Computes the xi coefficients to compute the worst-case delay bounds in a forest topology

        :param flows_interest: list of flows of interests
        :type flows_interest: list
        :param destination: root of the tree under analysis
        :type destination: int
        :return: a matrix of xi's
        :rtype: np.ndarray

        >>> tree = ToyTreeNetwork()
        >>> tree_exact = ExactFeedForwardAnalyzer(tree)
        >>> tree_exact.exact_xi([0], 2)
        array([[0.66666667, 0.66666667, 0.66666667],
               [0.        , 0.5       , 0.5       ],
               [0.        , 0.        , 0.5       ]])
        """
        if self.network.is_forest:
            xi = np.zeros((self.network.num_servers, self.network.num_servers))
            j = destination
            xi[j, j] = self.rstar(j, flows_interest) / (self.network.servers[j].scurve.rate -
                                                        self.xi_rate(flows_interest, j, j))
            list_suc = self.network.predecessors[j]
            while not list_suc == []:
                j = list_suc[0]
                r1 = self.rstar(j, flows_interest)
                r2 = self.network.servers[j].scurve.rate
                path_d = self.network.path_dest(j, destination)
                r2 -= sum(self.xi_rate(flows_interest, j, l) for l in path_d)
                k = destination
                while xi[self.network.successors[j], k] > r1 / r2:
                    xi[j, k] = xi[self.network.successors[j], k]
                    r2 += self.xi_rate(flows_interest, j, k)
                    r1 += xi[self.network.successors[j], k] * self.xi_rate(flows_interest, j, k)
                    path_d = path_d[1:]
                    k = path_d[0]
                for k in path_d:
                    xi[j, k] = r1 / r2
                xi[j, j] = r1 / r2
                list_suc = list_suc[1:] + self.network.predecessors[j]
            return xi
        else:
            raise NameError("Network is not a forest, exact analysis impossible")

    def latency_term(self, foi, server, xi):
        """
        Computes the term of the exact performance involving latencies. xi is the matrix exact_xi precomputed
        """
        lat = 0.
        for j in range(server+1):
            if j in self.network.path_dest(j, server):
                lat += (sum(xi[j, l] * self.xi_rate(foi, j, l) for l in self.network.path_dest(j, server))) * \
                    self.network.servers[j].scurve.latency
            if j in self.network.path_dest(j, server):
                lat += self.rstar(j, foi) * self.network.servers[j].scurve.latency
        return lat 
        
    def backlog(self, flow, server):
        """
        Computes a worst-case backlog bound of a flow in a server in the network with the Exact method for trees

        :param flow: flow under consideration
        :type flow: int
        :param server: server under consideration
        :type server: int
        :return: a backlog upper bound for the flow at the server
        :rtype: float

        >>> tree = ToyTreeNetwork()
        >>> tree_exact = ExactFeedForwardAnalyzer(tree)
        >>> tree_exact.backlog(0, 2)
        23.5
        """
        tnet = self.network.trim(server)
        ffa = ExactFeedForwardAnalyzer(tnet)
        xi = ffa.exact_xi([flow], server)
        b = 0
        for i in range(tnet.num_flows):
            if i == flow:
                b += tnet.flows[i].acurve.sigma
            else:
                if not tnet.flows[i].path == []:
                    b += xi[tnet.flows[i].path[0], tnet.flows[i].path[-1]] * \
                         tnet.flows[i].acurve.sigma
        b += self.latency_term([flow], server, xi)
        return b

    def delay(self, flow):
        start = self.network.flows[flow].path[0]
        end = self.network.flows[flow].path[-1]
        tnet = self.network.trim(end)
        ffa = ExactFeedForwardAnalyzer(tnet)
        xi = ffa.exact_xi([flow], end)
        bkl = self.backlog(flow, end)
        return (bkl - self.network.flows[flow].acurve.sigma * (1 - xi[start, end])) / self.network.flows[flow].acurve.rho
