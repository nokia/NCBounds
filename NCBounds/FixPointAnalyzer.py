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

from NCBounds.FeedForwardAnalyzer import *
from NCBounds.Flow import *
from NCBounds.Network import Network, Ring
from NCBounds.Server import Server
import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import linprog

solvers.options['show_progress'] = False


def resoud(mat_a, vec_b) -> np.ndarray:
    """
    solves the equation :math:` mat_a X = vec_b`

    :param mat_a: matrix of the equation
    :type mat_a: np.ndarray
    :param vec_b: vector of the equation
    :type vec_b: np.ndarray
    :return: the solution of the linear equation
    :rtype: np.ndarray
    """
    det = np.linalg.det(mat_a)
    if det != 0:
        return np.linalg.solve(mat_a, vec_b)
    else:
        return np.inf * np.ones(len(vec_b))


class FixPointAnalyzer:
    """
    Constructor for the fix point analyzer of a network. This network might contain cycles.
    This class regroup several subclasses, depending of the type of analyze to perform.
    Basically, it uses the Feed Forward analysis to obtain a fix-point equation to be solved. The different methods
    correspond to the different ways to obtain the equation.

    * SFA: Uses the SFA Feed-forwarw analysis
    * Exact, use the exact bounds after decomposing the network into a forest
    * Group uses the exact bounds applied to groups of flows along the arcs that have neem removed from the network
    * Combi combines the last two methods (using some mathematical programming)

    :param network: the network to analyze
    :type network: Network
    """
    def __init__(self, network):
        self.network = network

    @property
    def fixpoint_matrix(self):
        r"""
        Compute the fix-point matrix to solve, reprensented by the tuple (mat_a, vec_b), based on the chosen analysis.

        :return: the matrix and the vector such that :math:`mat_a \sigma = vec_b`
        :rtype: tuple
        """
        raise NotImplemented

    @property
    def ff_equiv(self) -> Network:
        r"""
        transforms a non feed-forward network into a feed-forward network by splitting the flows and computing the
            arrival curve of every splitted flow by the fixpoint method

        :return: a feed-forward network "equivalent" to the non feed-forward one.
        :rtype: Network
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


class SFAFixPointAnalyzer(FixPointAnalyzer):
    @property
    def fixpoint_matrix(self):
        r"""
        Compute the fix-point matrix to solve with the SFA method, reprensented by the tuple (mat_a, vec_b).
        For example, with the blind multiplexing, the  :math:`\sigma` after a server crossed by :math:`n` flows is
        given by:

        .. math:: \sigma'_1 = \sigma_1 + \rho_1\frac{\sum_{i=2}^n \sigma_i + RT}{R-\sum_{i=2}^n rho_i}.


        Computing these relations for all server and all flows crossing that server will lead to a system of linear
        equation, whise unknown are the :math:`\sigma`'s.

        :return: the matrix and the vector such that :math:`mat_a \sigma = vec_b`
        :rtype: tuple


        >>> toy = SFAFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.fixpoint_matrix
        (array([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [-1.  ,  1.  ,  0.  ,  0.  ,  0.  , -0.25,  0.  , -0.25,  0.  ],
               [ 0.  , -1.  ,  1.  , -0.25,  0.  ,  0.  ,  0.  ,  0.  , -0.25],
               [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  , -0.25,  0.  , -1.  ,  1.  ,  0.  ,  0.  ,  0.  , -0.25],
               [ 0.  ,  0.  , -0.25,  0.  , -1.  ,  1.  , -0.25,  0.  ,  0.  ],
               [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ],
               [ 0.  ,  0.  , -0.25,  0.  , -0.25,  0.  , -1.  ,  1.  ,  0.  ],
               [-0.25,  0.  ,  0.  ,  0.  ,  0.  , -0.25,  0.  , -1.  ,  1.  ]]), array([2., 6., 6., 2., 6., 6., 2., 6., 6.]))
        """
        s = self.network.size
        flo = self.network.matrix_topology
        res = self.network.residual_rate
        index = self.network.index
        mat_a = np.zeros((s, s))
        vec_b = np.zeros(s)
        for i in range(self.network.num_flows):
            for j in range(self.network.flows[i].length):
                path = self.network.flows[i].path
                idx = index[i, path[j]]
                if flo[i][path[j]] == 1:
                    vec_b[idx] = self.network.flows[i].acurve.sigma
                    mat_a[idx, idx] = 1.
                else:
                    mat_a[idx, idx] = 1.
                    mat_a[idx, index[i, path[j - 1]]] = -1.
                    for ic in range(self.network.num_flows):
                        if not (flo[ic, path[j - 1]] == 0. or ic == i):
                            mat_a[idx, index[ic, path[j - 1]]] = -self.network.flows[i].acurve.rho / res[i][path[j - 1]]
                    vec_b[idx] = self.network.flows[i].acurve.rho \
                        * self.network.servers[path[j - 1]].scurve.rate \
                        * self.network.servers[path[j - 1]].scurve.latency / res[i][path[j - 1]]
        return mat_a, vec_b

    @property
    def ff_equiv(self) -> Network:
        """
        transforms a non feed-forward network into a feed-forward network by splitting the flows and computing the
        arrival curve of every splitted flow by the fixpoint method with SFA method

        :return: The equivalent network
        :rtype: Network

        >>> toy = SFAFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.ff_equiv
        <Network:
        Flows:
              0:α(t) = 2.00 + 1.00 t; π = [0]
              1:α(t) = 24.40 + 1.00 t; π = [1]
              2:α(t) = 41.20 + 1.00 t; π = [2]
              3:α(t) = 2.00 + 1.00 t; π = [1]
              4:α(t) = 24.40 + 1.00 t; π = [2]
              5:α(t) = 41.20 + 1.00 t; π = [0]
              6:α(t) = 2.00 + 1.00 t; π = [2]
              7:α(t) = 24.40 + 1.00 t; π = [0]
              8:α(t) = 41.20 + 1.00 t; π = [1]
        Servers:
              0:β(t) = 6.00 . (t - 4.00)+
              1:β(t) = 6.00 . (t - 4.00)+
              2:β(t) = 6.00 . (t - 4.00)+>
        """
        tab_sigma = resoud(self.fixpoint_matrix[0], self.fixpoint_matrix[1])
        num_split = [self.network.flows[i].length for i in range(self.network.num_flows)]
        s = self.network.size
        list_flows = []
        j = 0
        h = 0
        flow = self.network.flows[j]
        for i in range(s):
            if tab_sigma[i] >= 0:
                list_flows += [Flow(ArrivalCurve(tab_sigma[i], flow.acurve.rho), [flow.path[h]])]
            else:
                list_flows += [Flow(ArrivalCurve(np.inf, flow.acurve.rho), [flow.path[h]])]
            h += 1
            if h == num_split[j]:
                h = 0
                j += 1
                if j < self.network.num_flows:
                    flow = self.network.flows[j]
        return Network(list_flows, self.network.servers)

    def _flow_decomp(self, flow, server):
        i = 0
        f = 0
        while i < flow:
            f += self.network.flows[i].length
            i += 1
        i = 0
        while not self.network.flows[flow].path[i] == server:
            f += 1
            i += 1
        return f

    def backlog(self, flow, server):
        """
        Computes a backlog bound of a flow at a server based on the SFA analysis.

        :param flow: flow for which the backlog is computed
        :type flow: int
        :param server: server at which the backlog is computed
        :type server: int
        :return: the backlog of flow and server
        :rtype: float

        >>> toy = SFAFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.backlog(0, 2)
        53.8
        """
        f = self._flow_decomp(flow, server)
        return SFAFeedForwardAnalyzer(self.ff_equiv).backlog(f, server)

    def delay(self, flow):
        """
        Computes a delay bound of a flow at a server based on the SFA analysis.

        :param flow: flow for which thedelay is computed
        :type flow: int
        :return: the delay of flow
        :rtype: float
        """
        ffnet = SFAFeedForwardAnalyzer(self.ff_equiv)
        tab_ac, tab_sc = ffnet.sfa_blind
        i = 0
        f = 0
        while i < flow:
            f += self.network.flows[i].length
            i += 1
        sc = ServiceCurve(np.inf, 0)
        for i in range(len(self.network.flows[flow].path)):
            sc = convolution(sc,
                             tab_sc[self.network.flows[flow].path[i] +
                                    (f + i) * self.network.num_servers])
        #server = self.network.flows[flow].path[-1]
        #f = self._flow_decomp(flow, server)
        #return SFAFeedForwardAnalyzer(self.ff_equiv).delay(f)
        return delay(self.network.flows[flow].acurve, sc)

class ExactFixPointAnalyzer(FixPointAnalyzer):
    @property
    def succ_forest(self):
        sf = np.zeros(self.network.num_servers, int)
        for i in range(self.network.num_servers):
            j = self.network.num_servers
            for k in self.network.successors[i]:
                if k > i:
                    j = min(k, j)
            sf[i] = j
        return sf

    @property
    def nk2forest(self) -> tuple:
        """
        Transforms the network into a forest by keeping one successor in the acyclic transformation: the one with
            the smallest number higher than this server

        :return:  network with split flows so that the topology is a forest and the list of the number of the flow
            before first split for each flow
        :rtype: tuple

        >>> toy = ExactFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.nk2forest
        (<Network:
        Flows:
              0:α(t) = 2.00 + 1.00 t; π = [0, 1, 2]
              1:α(t) = 2.00 + 1.00 t; π = [1, 2]
              2:α(t) = 2.00 + 1.00 t; π = [0]
              3:α(t) = 2.00 + 1.00 t; π = [2]
              4:α(t) = 2.00 + 1.00 t; π = [0, 1]
        Servers:
              0:β(t) = 6.00 . (t - 4.00)+
              1:β(t) = 6.00 . (t - 4.00)+
              2:β(t) = 6.00 . (t - 4.00)+>, [0, 1, 3])
        """
        flow_list = []
        list_prems = []
        pre = 0
        for flow in self.network.flows:
            i = 0
            list_prems += [pre]
            p = [flow.path[i]]
            while i < len(flow.path) - 1:
                if flow.path[i + 1] == self.succ_forest[flow.path[i]]:
                    p += [flow.path[i + 1]]
                else:
                    pre += 1
                    flow_list += [Flow(flow.acurve, p)]
                    p = [flow.path[i + 1]]
                i += 1
            pre += 1
            flow_list += [Flow(flow.acurve, p)]
        return Network(flow_list, self.network.servers), list_prems

    @property
    def fixpoint_matrix(self) -> tuple:
        r"""
        Compute the fix-point matrix to solve with the Rxact method, represented by the tuple (mat_a, vec_b).
        This make use of the matrix computing the :math:`xi` coefficients. The unknown are the :math:`\sigma` of the
        flows in the network transformed into a forest.

        :return: the matrix and the vector such that :math:`mat_a \sigma = vec_b`
        :rtype: tuple

        >>> toy = ExactFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.fixpoint_matrix
        (array([[ 1.  ,  0.  ,  0.  ,  0.  ,  0.  ],
               [ 0.  ,  1.  ,  0.  ,  0.  ,  0.  ],
               [-0.25, -1.  ,  0.9 , -0.25, -0.25],
               [ 0.  ,  0.  ,  0.  ,  1.  ,  0.  ],
               [-0.25, -0.25, -0.07, -1.  ,  0.9 ]]), array([ 2.  ,  2.  , 14.4 ,  2.  , 10.08]))
        """
        forest, list_prems = self.nk2forest
        s = len(forest.flows)
        mat_a = np.zeros((s, s))
        vec_b = np.zeros(s)
        list_prems += [forest.num_flows]
        i = 0
        for h in range(s):
            if h == list_prems[i]:
                vec_b[h] = forest.flows[i].acurve.sigma
                mat_a[h, h] = 1.
                i += 1
            else:
                ftrim = forest.trim(forest.flows[h-1].path[-1])
                # print (ftrim)
                ffa = ExactFeedForwardAnalyzer(ftrim)
                xi = ffa.exact_xi([h-1], forest.flows[h-1].path[-1])
                # print(xi)
                mat_a[h, h] = 1.
                mat_a[h, h - 1] = -1
                for h1 in range(s):
                    if not h - 1 == h1 and not ftrim.flows[h1].path == []:
                        mat_a[h, h1] -= xi[ftrim.flows[h1].path[0], ftrim.flows[h1].path[-1]]
                vec_b[h] = ffa.latency_term([h-1], forest.flows[h-1].path[-1], xi)
        return mat_a, vec_b

    @property
    def ff_equiv(self) -> Network:
        """
        transforms a non feed-forward network into a feed-forward network by splitting the flows and computing the
        arrival curve of every splitted flow by the fixpoint method with exact method

        :return: The equivalent network
        :rtype: Network

        >>> toy = ExactFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.ff_equiv
        <Network:
        Flows:
              0:α(t) = 2.00 + 1.00 t; π = [0, 1, 2]
              1:α(t) = 2.00 + 1.00 t; π = [1, 2]
              2:α(t) = 23.89 + 1.00 t; π = [0]
              3:α(t) = 2.00 + 1.00 t; π = [2]
              4:α(t) = 16.39 + 1.00 t; π = [0, 1]
        Servers:
              0:β(t) = 6.00 . (t - 4.00)+
              1:β(t) = 6.00 . (t - 4.00)+
              2:β(t) = 6.00 . (t - 4.00)+>
        """
        tab_sigma = resoud(self.fixpoint_matrix[0], self.fixpoint_matrix[1])
        forest = self.nk2forest[0]
        s = forest.num_flows
        list_flows = []
        for i in range(s):
            flow = forest.flows[i]
            if tab_sigma[i] >= 0:
                list_flows += [Flow(ArrivalCurve(tab_sigma[i], flow.acurve.rho), flow.path)]
            else:
                list_flows += [Flow(ArrivalCurve(np.inf, flow.acurve.rho), flow.path)]
        return Network(list_flows, self.network.servers)

    def _flow_decomp(self, flow, server):
        ff_net, list_prems = self.nk2forest
        f = list_prems[flow]
        if flow == self.network.num_flows - 1:
            b = ff_net.num_flows
        else:
            b = list_prems[flow + 1]
        while f < b and server not in ff_net.flows[f].path:
            f += 1
        if f == b:
            raise NameError("flow does not cross the server")
        return f

    def backlog(self, flow, server):
        """
        Computes a backlog bound of a flow at a server based on the exact analysis.

        :param flow: flow for which the backlog is computed
        :type flow: int
        :param server: server at which the backlog is computed
        :type server: int
        :return: the backlog of flow and server
        :rtype: float

        >>> toy = ExactFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.backlog(0, 2)
        31.069400630914828
        """
        f = self._flow_decomp(flow, server)
        # print(f)
        return ExactFeedForwardAnalyzer(self.ff_equiv).backlog(f, server)

    def delay(self, flow):
        """
        Computes a delay bound of a flow based on the exact analysis.
        WARNING: only for flows not cut into several subflows -> TODO

        :param flow: flow for which the delay is computed
        :type flow: int
        :return: the delay of flow
        :rtype: float
         """
        server = self.network.flows[flow].path[-1]
        f = self._flow_decomp(flow, server)
        # print(f)
        # print(ExactFeedForwardAnalyzer(self.ff_equiv).network)
        # print(f)
        return ExactFeedForwardAnalyzer(self.ff_equiv).delay(f)


class GroupFixPointAnalyzer(ExactFixPointAnalyzer):
    @property
    def _removed_edges(self) -> list:
        """
        Compute the set of edges that are removed when transforming the network into a forest.

        :return: the list of removed edges
        :rtype: list

        >>> toy = GroupFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy._removed_edges
        [(2, 0)]
        """
        lre = set([])
        for i in range(self.network.num_flows):
            for h in range(self.network.flows[i].length - 1):
                if not self.network.flows[i].path[h + 1] == self.succ_forest[self.network.flows[i].path[h]]:
                    lre.add((self.network.flows[i].path[h], self.network.flows[i].path[h + 1]))
        return list(lre)

    @property
    def foi_group(self):
        """
        For each removed edge, constructs the set of flows of interest for the analysis, that is the set of flows that
        were going through that edge. These will be the set of flows of interest for gurther analysis (we want the
        global worst-case backlog of these flows

        :return: the list of flows of interests for each removed edge
        :rtype: list

        >>> toy = GroupFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.foi_group
        [[1, 3]]
        """
        forest, list_prems = self.nk2forest
        list_prems += [forest.num_flows]
        list_per_edge = len(self._removed_edges) * [[]]
        for f in range(len(self._removed_edges)):
            (i, j) = self._removed_edges[f]
            s = 1
            for h in range(forest.num_flows - 1):
                if h + 1 == list_prems[s]:
                    s += 1
                elif (i, j) == (forest.flows[h].path[-1], forest.flows[h + 1].path[0]):
                    list_per_edge[f] = list_per_edge[f] + [h]
        return list_per_edge

    @property
    def fixpoint_matrix(self):
        r"""
        Compute the fix-point matrix to solve with the Exact method, represented by the tuple (mat_a, vec_b).
        This make use of the matrix computing the :math:`xi` coefficients. The unknown are the :math:`\sigma` of the
        grups of flows, per removed edge in the network transformed into a forest.

        :return: the matrix and the vector such that :math:`mat_a \sigma = vec_b`
        :rtype: tuple

        >>> toy = GroupFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.fixpoint_matrix
        (array([[0.72]]), array([18.]))
        """
        forest, list_prems = self.nk2forest
        redges = self._removed_edges
        rlist = self.foi_group
        #print(rlist)
        #print(list_prems)
        s = len(redges)
        mat_a = np.zeros((s, s))
        vec_b = np.zeros(s)
        for h in range(s):
            tforest = forest.trim(redges[h][0])
            ffa = ExactFeedForwardAnalyzer(tforest)
            xi = ffa.exact_xi(rlist[h], redges[h][0])
            # print(xi)
            mat_a[h, h] = 1
            for e in range(s):
                mat_a[h, e] -= max([0] + [xi[tforest.flows[f + 1].path[0],
                                             tforest.flows[f + 1].path[-1]]
                                          for f in rlist[e]])
                vec_b[h] = sum([xi[tforest.flows[f].path[0], tforest.flows[f].path[-1]]
                                * tforest.flows[f].acurve.sigma
                                for f in list_prems if not tforest.flows[f].path == []
                                and f not in rlist[h]])
                vec_b[h] += sum([tforest.flows[f].acurve.sigma
                                for f in list_prems if not tforest.flows[f].path == []
                                and f in rlist[h]])
                vec_b += ffa.latency_term(rlist[h], redges[h][0], xi)
            #print(mat_a, vec_b)
        return mat_a, vec_b

    @property
    def ff_equiv(self) -> Network:
        """
        transforms a non feed-forward network into a feed-forward network by splitting the flows and computing the
        arrival curve of every splitted flow by the fixpoint method with exact method and grouping flows.

        :return: The equivalent network
        :rtype: Network

        >>> toy = GroupFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.ff_equiv
        <Network:
        Flows:
              0:α(t) = 2.00 + 1.00 t; π = [0, 1, 2]
              1:α(t) = 2.00 + 1.00 t; π = [1, 2]
              2:α(t) = 30.53 + 1.00 t; π = [0]
              3:α(t) = 2.00 + 1.00 t; π = [2]
              4:α(t) = 30.53 + 1.00 t; π = [0, 1]
        Servers:
              0:β(t) = 6.00 . (t - 4.00)+
              1:β(t) = 6.00 . (t - 4.00)+
              2:β(t) = 6.00 . (t - 4.00)+>
        """
        tab_sigma = resoud(self.fixpoint_matrix[0], self.fixpoint_matrix[1])
        forest, list_prems = self.nk2forest
        s = forest.num_flows
        r = len(self._removed_edges)
        list_sigma = np.zeros(s)
        for i in range(self.network.num_flows):
            list_sigma[list_prems[i]] = self.network.flows[i].acurve.sigma
        for i in range(r):
            for f in self.foi_group[i]:
                if tab_sigma[i] >= 0:
                    list_sigma[f + 1] = tab_sigma[i]
                else:
                    list_sigma[f + 1] = np.inf
        list_flows = []
        for i in range(s):
            flow = forest.flows[i]
            list_flows += [Flow(ArrivalCurve(list_sigma[i], flow.acurve.rho), flow.path)]
        return Network(list_flows, self.network.servers)

    def backlog(self, flow, server):
        """
        Computes a backlog bound of a flow at a server based on the exact analysis.

        :param flow: flow for which the backlog is computed
        :type flow: int
        :param server: server at which the backlog is computed
        :type server: int
        :return: the backlog of flow and server
        :rtype: float

        >>> toy = GroupFixPointAnalyzer(Ring(3, ArrivalCurve(2., 1.), Server(ServiceCurve(6, 4))))
        >>> toy.backlog(0, 2)
        33.5
        """
        f = self._flow_decomp(flow, server)
        return ExactFeedForwardAnalyzer(self.ff_equiv).backlog(f, server)

    def delay(self, flow):
        """
        Computes a delay bound of a flow based on the exact analysis.
        WARNING: only for flows not cut into several subflows -> TODO

        :param flow: flow for which the delay is computed
        :type flow: int
        :return: the delay of flow
        :rtype: float
         """
        server = self.network.flows[flow].path[-1]
        f = self._flow_decomp(flow, server)
        #print(ExactFeedForwardAnalyzer(self.ff_equiv).network)
        # print(f)
        return ExactFeedForwardAnalyzer(self.ff_equiv).delay(f)



class LinearFixPointAnalyzer(GroupFixPointAnalyzer):
    @property
    def matrix_diag(self):
        (forest, prems) = self.nk2forest
        num_f = forest.num_flows
        removed_edges = self._removed_edges
        num_a = len(removed_edges)
        mat_up = np.eye(num_f)
        mat_bot = np.zeros((num_a, num_f))
        for i in range(num_a):
            for j in self.foi_group[i]:
                mat_bot[i, j+1] = 1
        return np.block([[mat_up], [mat_bot]])

    @property
    def matrix_end(self):
        (forest, prems) = self.nk2forest
        num_f = forest.num_flows
        removed_edges = self._removed_edges
        num_a = len(removed_edges)
        return -1 * np.eye(num_a + num_f)

    @property
    def fixpoint_matrix_flows(self) -> tuple:
        forest, list_prems = self.nk2forest
        s = len(forest.flows)
        mat_a = np.zeros((s, s))
        vec_b = np.zeros(s)
        list_prems += [forest.num_flows]
        i = 0
        for h in range(s):
            if h == list_prems[i]:
                vec_b[h] = forest.flows[i].acurve.sigma
                i += 1
            else:
                ftrim = forest.trim(forest.flows[h-1].path[-1])
                ffa = ExactFeedForwardAnalyzer(ftrim)
                xi = ffa.exact_xi([h-1], forest.flows[h-1].path[-1])
                mat_a[h, h - 1] = -1
                for h1 in range(s):
                    if not h - 1 == h1 and not ftrim.flows[h1].path == []:
                        mat_a[h, h1] -= xi[ftrim.flows[h1].path[0], ftrim.flows[h1].path[-1]]
                vec_b[h] = ffa.latency_term([h-1], forest.flows[h-1].path[-1], xi) 
        return mat_a, vec_b

    @property
    def fixpoint_matrix_arcs(self):
        forest, list_prems = self.nk2forest
        num_f = forest.num_flows
        redges = self._removed_edges
        rlist = self.foi_group
        num_a = len(redges)
        mat_a = np.zeros((num_a, num_f))
        vec_b = np.zeros(num_a)
        for h in range(num_a):
            tforest = forest.trim(redges[h][0])
            ffa = ExactFeedForwardAnalyzer(tforest)
            xi = ffa.exact_xi(rlist[h], redges[h][0])
            for e in range(num_f):
                if e in rlist[h]:
                    mat_a[h, e] = -1
                elif not tforest.flows[h].path == []:
                    mat_a[h, e] = -xi[tforest.flows[e].path[0], tforest.flows[e].path[-1]]
                vec_b[h] = ffa.latency_term(rlist[h], redges[h][0], xi)
        return mat_a, vec_b

    @property
    def matrix_f_and_a(self):
        (forest, prems) = self.nk2forest
        num_f = forest.num_flows
        removed_edges = self._removed_edges
        num_a = len(removed_edges)
        return np.block([[self.fixpoint_matrix_flows[0]], [self.fixpoint_matrix_arcs[0]]]), \
               np.block([[self.fixpoint_matrix_flows[1], self.fixpoint_matrix_arcs[1]]])

    def the_big_matrix_and_vector(self):
        mat_a, vec_b = self.matrix_f_and_a
        matrix_diag = self.matrix_diag
        matrix_end = self.matrix_end
        n1, n2 = np.shape(matrix_diag)
        mat0 = np.zeros((n1 + 1, n2))
        end_mat = np.eye(n1)
        big_a = np.block([[np.block([[mat_a[0]],[matrix_diag]]),
                       np.block([[mat0] * (n1 - 1)]),
                       np.block([[end_mat[0]], [matrix_end]])]])
        big_b = np.zeros(n1 + n1 * (n1 + 1))
        big_b[n1] = vec_b[0,0]
        for i in range(n1 - 2):
            big_b[n1 + (i + 1) * (n1+1)] = vec_b[0,i + 1]
            a_next = np.block([np.block([[mat0] * (i+1)]),
                          np.block([[mat_a[i+1]],[matrix_diag]]),
                          np.block([[mat0] * (n1 - i - 2)]),
                          np.block([[end_mat[i+1]], [matrix_end]])])
            big_a = np.concatenate((big_a, a_next), axis=0)
        a_next = np.block([[np.block([[mat0] * (n1 - 1)]),
                       np.block([[mat_a[n1 - 1]],[matrix_diag]]),
                       np.block([[end_mat[n1 - 1]], [matrix_end]])]])
        big_a = np.concatenate((big_a, a_next), axis = 0)
        big_b[n1 + (n1 - 1) * (n1+1) ] = vec_b[0,n1 - 1]
        big_a = np.block([[matrix_diag, np.zeros((n1, (n1 * n2))), matrix_end], [np.zeros((n1 * (n1 + 1), n2)), big_a]])
        return big_a, big_b

    def backlog(self, flow, server):
        mat_a, vec_b = self.the_big_matrix_and_vector()
        n1, n2 = np.shape(mat_a)
        tforest = self.nk2forest[0].trim(server)
        tfa = ExactFeedForwardAnalyzer(tforest)
        c = np.zeros(n2)
        f = self._flow_decomp(flow, server)
        xi = tfa.exact_xi([f], server)
        s = len(tforest.flows)
        for i in range(s):
            if i == f:
                c[i] = -1
            else:
                if not tforest.flows[i].path == []:
                    c[i] = -xi[tfa.network.flows[i].path[0], tfa.network.flows[i].path[-1]]
        linear = linprog(c, mat_a, vec_b, options={'tol':1e-7})
        if linear.success == True:
            bkl = -linear.fun
            bkl += tfa.latency_term([f], server, xi)
        else:
            bkl = np.inf
        return bkl

    def delay(self, flow):
        server = self.network.flows[flow].path[-1]
        start = self.network.flows[flow].path[0]
        mat_a, vec_b = self.the_big_matrix_and_vector()
        n1, n2 = np.shape(mat_a)
        tforest = self.nk2forest[0].trim(server)
        tfa = ExactFeedForwardAnalyzer(tforest)
        c = np.zeros(n2)
        f = self._flow_decomp(flow, server)
        xi = tfa.exact_xi([f], server)
        s = len(tforest.flows)
        for i in range(s):
            if i == f:
                c[i] = -1
            else:
                if not tforest.flows[i].path == []:
                    c[i] = -xi[tfa.network.flows[i].path[0], tfa.network.flows[i].path[-1]]
        linear = linprog(c, mat_a, vec_b, options={'tol':1e-7})
        if linear.success == True:
            bkl = -linear.fun
            bkl += tfa.latency_term([f], server, xi)
        else:
            bkl = np.inf
            return np.inf
        return (bkl - self.network.flows[flow].acurve.sigma * (1 - xi[start, server])) / self.network.flows[flow].acurve.rho
