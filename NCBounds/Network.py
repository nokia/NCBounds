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

import numpy as np
from NCBounds.Flow import *
from NCBounds.Server import *
from NCBounds.ArrivalCurve import ArrivalCurve
from NCBounds.Server import Server


def list_to_str(l: list) -> str:
    return "\n".join(["   %4i:%s" % (i, x) for (i, x) in enumerate(l)])


def print_list(l: list):
    print(list_to_str(l))


def dfs(u, n, a, state, queue, sort) -> tuple:
    """
    Depth-first-search implementation of a graph without cycles
    Args:
        u: source node
        n: sizeof the networks
        a: adjacency matric of the graph
        state: state of the nodes ('white'/'gray'/'black')
        queue: set of nodes queued for analysis
        sort: list of nodes in the reversed order of end of discovery (when they become 'black')

    Returns: the new state after exporation from u, new queue, update order

    """
    state[u] = 1
    queue = [u] + queue
    while not queue == []:
        for v in [x for x in range(n) if a[u][x] == 1]:
            if state[v] == 0:
                queue = [v] + queue
                state[v] = 1
                (state, queue, sort) = dfs(v, n, a, state, queue, sort)
            elif state[v] == 1:
                raise NameError("Network has cycles: feed-forward analysis impossible")
        sort = [u] + sort
        state[u] = 2
        queue = queue[1:]
        return state, queue, sort


def topological_sort(adjacency_matrix) -> list:
    """
    Topological sort of a graph given by its adjacency matrix
    Args:
        adjacency_matrix: adjacency matric of the graph

    Returns: the topological order of the nodes, and an error if the graph has cycles.

    """
    n, m = adjacency_matrix.shape
    if not (n == m):
        raise NameError("Adjacency matrix is not a square matrix")
    else:
        sort = []
        state = np.zeros(n, int)
        u = np.argmin(state)
        while state[u] == 0:
            (state, queue, sort) = dfs(u, n, adjacency_matrix, state, [], sort)
            u = np.argmin(state)
        return sort


class Network:
    """
    Constructor for a network described by a list of flows and a list of servers

    :param flows: list of flows circulating in the network
    :type flows: List[Flow]
    :param servers: list of servers of the network
    :type servers: List[Server]

    >>> flows = [Flow(ArrivalCurve(2, 1), [0,1 ]), Flow(ArrivalCurve(3, 2), [1, 0])]
    >>> servers = [Server(ServiceCurve(5, 1)), Server(ServiceCurve(6, 2))]
    >>> network = Network(flows, servers)
    >>> network
    <Network:
    Flows:
          0:α(t) = 2.00 + 1.00 t; π = [0, 1]
          1:α(t) = 3.00 + 2.00 t; π = [1, 0]
    Servers:
          0:β(t) = 5.00 . (t - 1.00)+
          1:β(t) = 6.00 . (t - 2.00)+>
    >>> network.num_flows
    2
    >>> network.num_servers
    2
    >>> network.size
    4

    >>> toy = ToyNetwork()
    >>> toy
    <Network:
    Flows:
          0:α(t) = 1.00 + 2.00 t; π = [0, 1, 2]
          1:α(t) = 2.00 + 3.00 t; π = [1, 2]
          2:α(t) = 1.00 + 1.00 t; π = [0, 2]
    Servers:
          0:β(t) = 4.00 . (t - 2.00)+
          1:β(t) = 5.00 . (t - 3.00)+
          2:β(t) = 7.00 . (t - 1.00)+>
    >>> toy.num_flows
    3
    >>> toy.num_servers
    3
    >>> toy.size
    7
    """
    def __init__(self, flows=list(), servers=list()):
        self.flows = flows
        self.servers = servers
        self.num_flows = len(flows)
        self.num_servers = len(servers)
        self.size = sum(self.flows[i].length for i in range(self.num_flows))
        self._flows_in_servers = None
        self._adjacency_matrix = None

    def __str__(self) -> str:
        return "Flows:\n%s\nServers:\n%s" % (list_to_str(self.flows), list_to_str(self.servers))

    def __repr__(self) -> str:
        return "<Network:\n%s>" % self.__str__()

    @property
    def index(self) -> np.ndarray:
        mat = np.ones((self.num_flows, self.num_servers), dtype=int)
        mat = -mat
        counter = 0
        for i in range(self.num_flows):
            for j in range(self.flows[i].length):
                mat[i][self.flows[i].path[j]] = counter
                counter += 1
        return mat

    @property
    def matrix_topology(self) -> np.ndarray:
        """
        Computes :math:`M` the matrix topology. The matrix has :attr:`num_flows` lines and :attr:`num_servers` columns,
        and :math:`M_{i,j} = k` if server :math:`j` is the :math:`k`-th server crossed by flow :math:`i`.

        :return: a matrix as defined above
        :rtype: np.ndarray

        >>> toy = ToyNetwork()
        >>> toy.matrix_topology
        array([[1, 2, 3],
               [0, 1, 2],
               [1, 0, 2]])
        """
        mat = np.zeros((self.num_flows, self.num_servers), int)
        for i in range(self.num_flows):
            pf = self.flows[i].path
            for j in range(len(pf)):
                mat[i, pf[j]] = j + 1
        return mat

    @property
    def flows_in_servers(self) -> list:
        r"""
        Returns the list of the list of the flows crossing each server:

        .. math:: FiS[j] = \{i~|~j\in\pi_i\}

        :return: the list of the list of the flows crossing each server
        :rtype: list

        >>> toy = ToyNetwork()
        >>> toy.flows_in_servers
        [[0, 2], [0, 1], [0, 1, 2]]
        """
        if self._flows_in_servers is None:
            fis = self.num_servers * [[]]
            for i in range(self.num_flows):
                for p in self.flows[i].path:
                    fis[p] = fis[p] + [i]
            self._flows_in_servers = fis
        return self._flows_in_servers

    @property
    def residual_rate(self) -> np.ndarray:
        fis = self.flows_in_servers
        mat = np.zeros((self.num_flows, self.num_servers), int)
        for j in range(self.num_servers):
            res_rate = self.servers[j].scurve.rate - sum(self.flows[i].acurve.rho for i in fis[j])
            for i in fis[j]:
                mat[i][j] = res_rate + self.flows[i].acurve.rho
        return mat

    @property
    def adjacency_matrix(self) -> np.ndarray:
        r"""
        Constructs the adjacency matrix :math:`A` of the network:

        .. math:: A[h,\ell] = 1 \text{ if }\exists i \text{ such that } \pi_i = \langle \ldots h, \ell, \ldots \rangle

        and 0 otherwise

        :return: the adjacency matrix of the network
        :rtype: np.ndarray

        >>> toy = ToyNetwork()
        >>> toy.adjacency_matrix
        array([[0, 1, 1],
               [0, 0, 1],
               [0, 0, 0]])
        """
        if self._adjacency_matrix is None:
            n = self.num_servers
            adj = np.zeros((n, n), int)
            for i in range(self.num_flows):
                flow = self.flows[i]
                for j in range(self.flows[i].length - 1):
                    adj[flow.path[j], flow.path[j+1]] = 1
            self._adjacency_matrix = adj
        return self._adjacency_matrix

    @property
    def successors(self) -> list:
        r"""
        Constructs the list :math:`Succ` of servers successors of each server:

        .. math:: Succ[h] = \{\ell~|~A[h, \ell] = 1\}.

        :return: the list of the lists of successors
        :rtype: List(list)

        >>> toy = ToyNetwork()
        >>> toy.successors
        [[1, 2], [2], []]
        """
        succ = self.num_servers * [[]]
        for i in range(self.num_servers):
            succ[i] = [j for j in range(self.num_servers) if self.adjacency_matrix[i, j] == 1]
        return succ

    @property
    def predecessors(self) -> list:
        r"""
        Constructs the list :math:`Pred` of servers predecessors of each server:

        .. math:: Pred[\ell] = \{h~|~A[h, \ell] = 1\}.

        :return: the list of the lists of predecessors
        :rtype: List(list)

        >>> toy = ToyNetwork()
        >>> toy.predecessors
        [[], [0], [0, 1]]
        """
        prec = self.num_servers * [[]]
        for i in range(self.num_servers):
            prec[i] = [j for j in range(self.num_servers) if self.adjacency_matrix[j, i] == 1]
        return prec

    @property
    def is_forest(self) -> bool:
        """
        Checks that a feed-forward network has a forest topology, that is if each server has at most one successor

        Caveat: if the network has cycle dependencies, then the algorithme can return True

        :return: true if the network is a forest
        :rtype: bool


        >>> toy = ToyNetwork()
        >>> toy.is_forest
        False

        >>> tree = ToyTreeNetwork()
        >>> tree.is_forest
        True
        """
        for i in range(self.num_servers):
            if len(self.successors[i]) >= 2:
                return False
        return True

    def path_dest(self, j, dest) -> list:
        """
        In a tree (0r a forest), constructs the path from server j to server dest, given in reversed order..

        :param j: the server from where the path is computed
        :type j: int
        :param dest: the destination server
        :type dest: int
        :return: the list of servers from j to dest (including those servers) in the order [dest,...,j] and
         the empty list if there is no path from j to dest.
        :rtype: list

        >>> toy = ToyTreeNetwork()
        >>> toy.path_dest(0,2)
        [2, 1, 0]
        >>> toy.path_dest(2,1)
        []
        """
        if self.is_forest:
            p = [j]
            k = j
            while not (k == dest):
                if self.successors[k] == []:
                    return []
                else:
                    k = self.successors[k][0]
                    p = [k] + p
            return p
        else:
            raise NameError("not a forest topology, path might be undefined or not unique")

    def trim(self, server):
        """
        In a forest, remove all servers that have a larger number than server
        Caveat: Should be used for ordered forests

        :param server: server that will become a root of the forest
        :type server: int
        :return: the trimmed network
        :rtype: Network

        >>> tree = ToyTreeNetwork()
        >>> tree.trim(1)
        <Network:
        Flows:
              0:α(t) = 1.00 + 2.00 t; π = [0, 1]
              1:α(t) = 2.00 + 3.00 t; π = [1]
              2:α(t) = 1.00 + 1.00 t; π = [0, 1]
        Servers:
              0:β(t) = 4.00 . (t - 2.00)+
              1:β(t) = 8.00 . (t - 3.00)+>
        """
        list_servers = self.servers[0:server+1]
        list_path = self.num_flows * [[]]
        for i in range(self.num_flows):
            list_path[i] = [self.flows[i].path[p] for p in range(self.flows[i].length)
                            if self.flows[i].path[p] <= server]
        list_flows = [Flow(self.flows[i].acurve, list_path[i]) for i in range(self.num_flows)]
        return Network(list_flows, list_servers)


class Ring(Network):
    """
    Constructor for a ring network of size :math:`n` = :attr:`num_flows`:  here are :math:`n` flows of length :math:`n`,
    starting one at each server. Every server has the same service curve/policy and every flows has the same arrival
    curve, and server :math:`i` is the successor of server :math:`i+1`, server :math:`n-1` is the successor of
    server :math:`0`.

    :param num_servers: number of servers in the ring-network
    :type num_servers: int
    :param acurve: arrival curve common to every flow
    :type acurve: ArrivalCurve
    :param server: server description common to every server
    :type server: Server

    >>> Ring(3, ArrivalCurve(2., 3.), Server(ServiceCurve(3, 4)))
    <Network:
    Flows:
          0:α(t) = 2.00 + 3.00 t; π = [0 1 2]
          1:α(t) = 2.00 + 3.00 t; π = [1 2 0]
          2:α(t) = 2.00 + 3.00 t; π = [2 0 1]
    Servers:
          0:β(t) = 3.00 . (t - 4.00)+
          1:β(t) = 3.00 . (t - 4.00)+
          2:β(t) = 3.00 . (t - 4.00)+>
    """
    def __init__(self, num_servers, acurve, server):
        super(Ring, self).__init__(
            [Flow(acurve, np.concatenate([np.arange(i, num_servers), np.arange(0, i)])) for i in range(num_servers)],
            num_servers * [server]
        )


class ToyNetwork(Network):
    """
    Construction for a toy network used in the tests.

    >>> ToyNetwork()
    <Network:
    Flows:
          0:α(t) = 1.00 + 2.00 t; π = [0, 1, 2]
          1:α(t) = 2.00 + 3.00 t; π = [1, 2]
          2:α(t) = 1.00 + 1.00 t; π = [0, 2]
    Servers:
          0:β(t) = 4.00 . (t - 2.00)+
          1:β(t) = 5.00 . (t - 3.00)+
          2:β(t) = 7.00 . (t - 1.00)+>
    """
    def __init__(self):
        super(ToyNetwork, self).__init__([Flow(ArrivalCurve(1, 2), [0, 1, 2]), Flow(ArrivalCurve(2, 3), [1, 2]),
                                          Flow(ArrivalCurve(1, 1), [0, 2])],
                                         [Server(ServiceCurve(4, 2)), Server(ServiceCurve(5, 3)),
                                          Server(ServiceCurve(7, 1))])


class ToyTreeNetwork(Network):
    """
    Construction for a toy network used in the tests with a tree topology.

    >>> ToyTreeNetwork()
    <Network:
    Flows:
          0:α(t) = 1.00 + 2.00 t; π = [0, 1, 2]
          1:α(t) = 2.00 + 3.00 t; π = [1, 2]
          2:α(t) = 1.00 + 1.00 t; π = [0, 1]
    Servers:
          0:β(t) = 4.00 . (t - 2.00)+
          1:β(t) = 8.00 . (t - 3.00)+
          2:β(t) = 7.00 . (t - 1.00)+>
    """
    def __init__(self):
        super(ToyTreeNetwork, self).__init__([Flow(ArrivalCurve(1, 2), [0, 1, 2]), Flow(ArrivalCurve(2, 3), [1, 2]),
                                              Flow(ArrivalCurve(1, 1), [0, 1])],
                                             [Server(ServiceCurve(4, 2)), Server(ServiceCurve(8, 3)),
                                              Server(ServiceCurve(7, 1))])


class TwoRings(Network):
    def __init__(self, num_servers_per_ring, acurve, server1, server2):
        super(TwoRings, self).__init__([Flow(acurve, np.concatenate([np.arange(i, num_servers_per_ring - 1),
                                                                     [2 * num_servers_per_ring - 2], np.arange(0, i)]))
                                        for i in range(num_servers_per_ring)] +
                                       [Flow(acurve, np.concatenate([np.arange(num_servers_per_ring - 1 + i,
                                                                               2 * num_servers_per_ring - 2),
                                                                     [2 * num_servers_per_ring - 2],
                                                                     np.arange(num_servers_per_ring - 1,
                                                                               num_servers_per_ring - 1 + i)]))
                                        for i in range(num_servers_per_ring)],
                                       (2 * num_servers_per_ring - 2) * [server1] + [server2])
