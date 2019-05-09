from NCBounds.ArrivalCurve import ArrivalCurve
from NCBounds.Flow import Flow
from NCBounds.ServiceCurve import ServiceCurve
from NCBounds.Server import Server
from NCBounds.Network import Network, Ring, TwoRings
from NCBounds.FeedForwardAnalyzer import FeedForwardAnalyzer, SFAFeedForwardAnalyzer, ExactFeedForwardAnalyzer
from NCBounds.FixPointAnalyzer import FixPointAnalyzer, SFAFixPointAnalyzer, ExactFixPointAnalyzer, \
                                        GroupFixPointAnalyzer, LinearFixPointAnalyzer

class SmallNetwork(Network):
    def __init__(self, u):
        super(SmallNetwork, self).__init__([Flow(ArrivalCurve(1, 25 * u), [2, 3, 1]), 
                                        Flow(ArrivalCurve(1, 25 * u), [3, 1, 2]),
                                        Flow(ArrivalCurve(1, 25 * u), [1, 0, 2]),
                                        Flow(ArrivalCurve(1, 25 * u), [1, 2, 3])],
                                       [Server(ServiceCurve(100, 1)), Server(ServiceCurve(100, 1)),
                                        Server(ServiceCurve(100, 1)), Server(ServiceCurve(100, 1))])


snk = SmallNetwork(0.5)

print("** The small Network description **")

print(snk)


print("\n ** Transformation into a forest (noy yet with the correct arrival curves)**")

forest_snk = ExactFixPointAnalyzer(snk).nk2forest[0]
print(forest_snk)

print("\n The matrix of xi computed for flow 6 and server 3")
print(ExactFeedForwardAnalyzer(forest_snk).exact_xi([6],3))


print("\n \nComputing the equivalent forest network (with correct arrival curves) and performances")

print("\n\t*SFA method: each flow is decomposed into sub-paths of length 1")

sfa = SFAFixPointAnalyzer(snk)

print(sfa.ff_equiv)
print(sfa.backlog(3, 3))

print("\n\t*Linear-flows method: the network is decomposed into a tree, and an arrival curve is computed for each sub-path")

exact = ExactFixPointAnalyzer(snk)

print(exact.ff_equiv)
print(exact.backlog(3, 3))

print("\n\t*Linear-arc method: the network is decomposed into a tree, and an arrival curve is computed for each arc that has been removed")

group = GroupFixPointAnalyzer(snk)

print(group.ff_equiv)
print(group.backlog(3, 3))


print("\n\t*Linear method: the network is decomposed into a tree, and an arrival curve is computed for each arc that has been removed and each sub-path of the flows")

linear = LinearFixPointAnalyzer(snk)

#print(linear.ff_equiv)
print(linear.backlog(3, 3))

print("\n\nComparing the approaches")


f = open('./small_network.data', 'w')
#f.write("# u\t SFA\t Exact\t Group \t Comby\n")
u=0.01
while u < 1:
    snk = SmallNetwork(u)
    f.write("%f\t" % u)
    f.write("%f\t" % SFAFixPointAnalyzer(snk).backlog(3, 3))
    f.write("%f\t" % ExactFixPointAnalyzer(snk).backlog(3, 3))
    f.write("%f\t" % GroupFixPointAnalyzer(snk).backlog_bis(3, 3))
    f.write("%f\n" % LinearFixPointAnalyzer(snk).backlog(3, 3))
    u += 0.01
f.close()

with open('./small_network.data') as f:
    lines = f.readlines()
    u = [float(line.split()[0]) for line in lines]
    sfa = [float(line.split()[1]) for line in lines]
    exact = [float(line.split()[2]) for line in lines]
    group = [float(line.split()[3]) for line in lines]
    combi = [float(line.split()[4]) for line in lines]
    
f.close()

import matplotlib.pyplot as pl
pl.plot(u,sfa, c='r', label='SFA')
pl.plot(u,exact, c='b', label='Flows')
pl.plot(u,group, c='y', label='Arcs')
pl.plot(u,combi, c='m', label='F+A')

pl.xlabel('Utilization rate')
pl.ylabel('Backlog bound')
pl.legend()
pl.axis([0., 1, 0, 2000])
pl.show()
