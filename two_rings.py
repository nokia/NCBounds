from NCBounds.ArrivalCurve import ArrivalCurve
from NCBounds.Flow import Flow
from NCBounds.ServiceCurve import ServiceCurve
from NCBounds.Server import Server
from NCBounds.Network import Network, Ring, TwoRings
from NCBounds.FeedForwardAnalyzer import FeedForwardAnalyzer, SFAFeedForwardAnalyzer, ExactFeedForwardAnalyzer
from NCBounds.FixPointAnalyzer import FixPointAnalyzer, SFAFixPointAnalyzer, ExactFixPointAnalyzer, \
                                        GroupFixPointAnalyzer, LinearFixPointAnalyzer
import numpy as np

n =5
R = 100
u = 0.3
d = np.inf
T = 0.001

two_ring = TwoRings(n, ArrivalCurve(1, u), Server(ServiceCurve(n, 1)), Server(ServiceCurve(2 * n, 1)))



print("** The small Network description **")

print(two_ring)


print("\n ** Transformation into a forest (noy yet with the correct arrival curves)**")

forest_two_ring = ExactFixPointAnalyzer(two_ring).nk2forest[0]
print(forest_two_ring)

print("\n The matrix of xi computed for flow 0 and server 4")
print(ExactFeedForwardAnalyzer(forest_two_ring).exact_xi([0],2 * (n - 1)))


print("\n \nComputing the equivalent forest network (with correct arrival curves) and performances")

print("\n\t*SFA method: each flow is decomposed into sub-paths of length 1")

sfa = SFAFixPointAnalyzer(two_ring)

print(sfa.ff_equiv)
print(sfa.backlog(0, 2*(n-1)))

print("\n\t*Linear-flows method: the network is decomposed into a tree, and an arrival curve is computed for each sub-path")

exact = ExactFixPointAnalyzer(two_ring)

print(exact.ff_equiv)
print(exact.backlog(0,2*(n-1 )))

print("\n\t*Linear-arc method: the network is decomposed into a tree, and an arrival curve is computed for each arc that has been removed")

group = GroupFixPointAnalyzer(two_ring)

print(group.ff_equiv)
#print(group.ff_equiv_bis)
#print(group.backlog_bis(0, 2*(n-1)))


print("\n\t*Linear method: the network is decomposed into a tree, and an arrival curve is computed for each arc that has been removed and each sub-path of the flows")

linear = LinearFixPointAnalyzer(two_ring)

# #print(linear.ff_equiv)
#print(linear.backlog(0, 2*(n-1)))

print("\n\nComparing the approaches")

k = 2* (n-1)
f = open('./two_ring_delay_3.data', 'w')
#f.write("# u\t SFA\t Exact\t Group \t Comby\n")
two_ring = TwoRings(n, ArrivalCurve(1, u), Server(ServiceCurve(n, T)), Server(ServiceCurve(2 * n, T)))
u = 0.01
lin = 0
while u < 1 and lin < 50000.:
    two_ring = TwoRings(n, ArrivalCurve(1, u), Server(ServiceCurve(n, T)), Server(ServiceCurve(2 * n, T)))
    f.write("%f\t" % u)
    f.write("%f\t" % SFAFixPointAnalyzer(two_ring).delay(0))
    f.write("%f\t" % ExactFixPointAnalyzer(two_ring).delay(0))
    f.write("%f\t" % GroupFixPointAnalyzer(two_ring).delay(0))
    lin = LinearFixPointAnalyzer(two_ring).delay(0)
    f.write("%f\n" % lin)
    print(u)
    u += 0.01
f.close()

# while u < 1 and lin < 50000.:
#     two_ring = TwoRings(n, ArrivalCurve(1, u), Server(ServiceCurve(n, T)), Server(ServiceCurve(2 * n, T)))
#     f.write("%f\t" % u)
#     f.write("%f\t" % SFAFixPointAnalyzer(two_ring).backlog(0, 2*(n-1 )))
#     f.write("%f\t" % ExactFixPointAnalyzer(two_ring).backlog(0, 2*(n-1 )))
#     f.write("%f\t" % GroupFixPointAnalyzer(two_ring).backlog(0, 2*(n-1 )))
#     lin = LinearFixPointAnalyzer(two_ring).backlog(0, 2*(n-1 ))
#     f.write("%f\n" % lin)
#     print(u)
#     u += 0.01
# f.close()

with open('./two_ring_delay_3.data') as f:
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
pl.axis([0., 1, 1, 200])
#pl.semilogy()
pl.show()
