import numpy as np
from NCBounds.ArrivalCurve import ArrivalCurve
from NCBounds.Flow import Flow
from NCBounds.ServiceCurve import ServiceCurve
from NCBounds.Server import Server
from NCBounds.Network import Network, Ring, TwoRings
from NCBounds.FeedForwardAnalyzer import FeedForwardAnalyzer, SFAFeedForwardAnalyzer, ExactFeedForwardAnalyzer
from NCBounds.FixPointAnalyzer import FixPointAnalyzer, SFAFixPointAnalyzer, ExactFixPointAnalyzer, \
                                        GroupFixPointAnalyzer, LinearFixPointAnalyzer


def backlogPMOO(n):
    """
    Computes the backlog bound of flow 0 at the last server with the PMOC method (based on PMOO tandem approach)
    """
    C = np.zeros((n-1,1))
    for i in range(n-1):
        C[i,0] = (i+1)*T + ((i+n-1)* b + (i+1)*(n-1)*r*T)/(R-(n-1)*r)
    a = r/(R-(n-1)*r)
    I = np.identity(n-1)
    U = np.ones((n-1,n-1))
    A = np.linalg.inv(I-a*U)
    matT = np.dot(A,C)
    return  b + r * (n*T + (2*(n-1)*b + n * (n-1) * r * T)/(R-(n-1)*r) + a * np.sum(matT) )

def delayPMOO(n):
    """
    Computes the delay bound of flow 0  with the PMOC method (based on PMOO tandem approach)
    """
    C = np.zeros((n-1,1))
    for i in range(n-1):
        C[i,0] = (i+1)*T + ((i+n-1)* b + (i+1)*(n-1)*r*T)/(R-(n-1)*r)
    a = r/(R-(n-1)*r)
    I = np.identity(n-1)
    U = np.ones((n-1,n-1))
    A = np.linalg.inv(I-a*U)
    matT = np.dot(A,C)
    return  (n*T + (2*(n-1)*b + n * (n-1) * r * T)/(R-(n-1)*r) + a * np.sum(matT) ) + b / (R - (n - 1) * r)

def backlogLeBoudec(n):
    """
    Computes the  backlog bound of flow 0 at the last server based on the results in Le Boudec, Thiran 2004.
    """
    return(n*r / (R - n*r) * (n * n + n*R*T)  + n * (1 + R*T))

n = 10
# r = 10
R = 100
u = 0.01
T = 0.001
b = 1
f = open('./uni_ring_%d.data' % n, 'w')
while u<1:
    r = u*R/n
    ring = Ring(n, ArrivalCurve(b, r), Server(ServiceCurve(R,T)))
    f.write("%f\t" % u)
    f.write("%f\t" % SFAFixPointAnalyzer(ring).backlog(0, n-1))
    if u< n/(2*(n-1)):
        f.write("%f\t" % backlogPMOO(n))
    else: 
        f.write("inf\t")
    f.write("%f\t" % ExactFixPointAnalyzer(ring).backlog(0, n-1))
    f.write("%f\t" % GroupFixPointAnalyzer(ring).backlog(0, n-1))
    f.write("%f\t" % LinearFixPointAnalyzer(ring).backlog(0, n-1))
    f.write("%f\n" % backlogLeBoudec(n))
    print(R, u)
    u += 0.01
f.close()

with open('./uni_ring_%d.data' % n) as f:
    lines = f.readlines()
    u = [float(line.split()[0]) for line in lines]
    sfa = [float(line.split()[1]) for line in lines]
    pmoo = [float(line.split()[2]) for line in lines]
    exact = [float(line.split()[3]) for line in lines]
    arc = [float(line.split()[4]) for line in lines]
    linear = [float(line.split()[5]) for line in lines]
    leboudec = [float(line.split()[6]) for line in lines]
f.close()

import matplotlib.pyplot as pl
pl.plot(u,sfa, c='r', label='SFA')
pl.plot(u,exact, c='b', label='Flows')
pl.plot(u,pmoo, c='y', label='PMOO')
pl.plot(u,arc, c='m', label='Arcs')
pl.plot(u,linear, c='r', label='F+A')
pl.plot(u,leboudec, c='g', label='Leboudec')
pl.xlabel('Utilization rate')
pl.ylabel('Delay bound')
pl.legend()

pl.axis([0, 1, 0, 100])
#pl.semilogy()
pl.show()
    
