import numpy as np
import miepython
import matplotlib.pyplot as plt

def hg(g,costheta):
    return (1/4/np.pi)*(1-g**2)/(1+g**2-2*g*costheta)**1.5

num=1000   # increase number of angles to improve integration
r=4.5  # in microns
lambdaa = 0.76 # in microns
m = complex(1.3057,7.08E-8)

x = 2*np.pi*r/lambdaa
k = 2*np.pi/lambdaa
qext, qsca, qback, g =  miepython.mie(m,x)

mu = np.linspace(-1,1,num)
s1,s2 =  miepython.mie_S1_S2(m,x,mu)
miescatter = 0.5*(abs(s1)**2+abs(s2)**2)
hgscatter = hg(0.8925954539701679,mu)

delta_mu=mu[1]-mu[0]
total = 2*np.pi*delta_mu*np.sum(miescatter)

print('mie integral= ',total)

total = 2*np.pi*delta_mu*np.sum(hgscatter)
print('hg integral= ', total)

plt.semilogy(mu,hgscatter)
plt.semilogy(mu,miescatter)
