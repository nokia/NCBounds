import numpy as np
from numpy import linalg as la

R = 100

def mat_exacte(n, R, u):
    r = u*R/n
    ρ = r / (R - (n-1) * r)
    x = np.zeros(n)
    x[0] = ρ
    for i in np.arange(1,n):
        x[i] = (x[0] * (n-i) * r + r * sum([x[j] for j in np.arange(1,i)])) / (R - r)
    return np.array([[x[max(i  + 1 - j,0)]for j in np.arange(n-1)] for i in np.arange(n-1)])

N = 100
tabEXACT_u = np.ones(N+1)
tab_n = np.arange(N+1)
u = 1
n = 2
b = True
while b :
    r = u*R/n
    ρ = r / (R- (n-1)*r)
    mat = mat_exacte(n, R, u)
    s = max(la.eigvals(mat))
    if s < 1:
        tabEXACT_u[n] = u
        n += 1
    else:
        u -= 0.0001
    if u < 0 or n > N:
        b = False


tabSFA_u = np.ones(N+1)
u = 1
n = 2
b = True
while b :
    r = u*R/n
    ρ = r / (R- (n-1)*r)
    mat = ρ * np.ones((n-1,n-1)) 
    for i in np.arange(n-2):
        mat[i+1,i] = 1
    s = max(la.eigvals(mat))
    if s < 1:
        tabSFA_u[n] = u
        n += 1
    else:
        u -= 0.001
    if u < 0 or n > N:
        b = False

tabPMOO_u = np.ones(N+1)
u = 1
n = 2
b = True
while b :
    r = u*R/n
    ρ = r / (R- (n-1)*r)
    mat = ρ * np.ones((n-1,n-1)) 
    s = max(la.eigvals(mat))
    if s < 1:
        tabPMOO_u[n] = u
        n += 1
    else:
        u -= 0.001
    if u < 0 or n > N:
        b = False

f = open('./ring_stability.data', 'w')
n = 0
while n<N+1:
    f.write("%f\t" % n)
    f.write("%f\t" % tabEXACT_u[n])
    f.write("%f\t" % tabPMOO_u[n])
    f.write("%f\n" % tabSFA_u[n])
    n += 1
f.close()

with open('./ring_stability.data') as f:
    lines = f.readlines()
    n = [float(line.split()[0]) for line in lines]
    exact = [float(line.split()[1]) for line in lines]
    pmoo = [float(line.split()[2]) for line in lines]
    sfa = [float(line.split()[3]) for line in lines]
    
f.close()

import matplotlib.pyplot as pl
pl.plot(n,sfa, c='r', label='SFA')
pl.plot(n,exact, c='b', label='Exact')
pl.plot(n,pmoo, c='y', label='PMOO')

pl.xlabel('Number of servers')
pl.ylabel('Maximum utilization rate for stability')
pl.legend()

pl.axis([0, 100, 0, 1])
pl.show()
