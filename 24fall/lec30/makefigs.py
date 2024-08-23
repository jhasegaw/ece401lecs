import numpy  as np
import matplotlib.figure, subprocess, os

os.makedirs('exp',exist_ok=True)
################################################################################
#   Picture showing $\left(a^nu[n]\right) + b\left(a^{n-1}u[n-1]\right)$
fig = matplotlib.figure.Figure((8,4))
axs = fig.subplots(3,1,sharex=True)
nset = np.arange(-3,12)
a = 0.85
b = 0.5
h1 = np.zeros(len(nset))
h1[nset>=0] = np.power(a,nset[nset>=0])
h2 = np.zeros(len(nset))
h2[nset>0] = b*np.power(a,nset[nset>0]-1)
axs[0].stem(nset,h1)
axs[0].set_title('$(0.85)^n u[n]$')
axs[0].set_ylim([-0.01,1.5])
axs[1].stem(nset,h2)
axs[1].set_title('$0.5(0.85)^{n-1} u[n-1]$')
axs[1].set_ylim([-0.01,1.5])
axs[2].stem(nset,h1+h2)
axs[2].set_title('$(0.85)^n u[n] + 0.5(0.85)^{n-1} u[n-1]$')
axs[2].set_ylim([-0.01,1.5])
fig.tight_layout()
fig.savefig('exp/numsum.png')

################################################################################
#   Image showing $g[n] = (0.5-0.5j)(0.6+0.6j)^nu[n]+(0.5+0.5j)(0.6-j0.6)^nu[n]$
fig = matplotlib.figure.Figure((8,4))
axs = fig.subplots(3,1,sharex=True)
nset = np.array([-3,-2,-1,-0.0001,0,1,2,3,4,5,6,7,8,9,10,11])
p1 = 0.6+0.6j
p2 = 0.6-0.6j
C1=0.5-0.5j
C2=0.5+0.5j
h1 = np.zeros(len(nset),dtype='complex')
h1[nset>=0] = C1*np.power(p1,nset[nset>=0])
h2 = np.zeros(len(nset),dtype='complex')
h2[nset>=0] = C2*np.power(p2,nset[nset>=0])
axs[0].stem(nset,np.real(h1))
axs[0].plot(nset,np.imag(h1),'b--')
axs[0].set_title('$g_1[n]=(0.5-0.5j)(0.6+0.6j)^n u[n]$ (imaginary part dashed)')
axs[0].set_ylim([-0.7,0.7])
axs[1].stem(nset,np.real(h2))
axs[1].plot(nset,np.imag(h2),'b--')
axs[1].set_title('$g_2[n]=(0.5+0.5j)(0.6-0.6j)^n u[n]$ (imaginary part dashed)')
axs[1].set_ylim([-0.7,0.7])
axs[2].stem(nset,np.real(h1+h2))
axs[2].set_title('$g_1[n]+g_2[n]=(0.5\sqrt{2})(0.6\sqrt{2})^ncos(\pi (n-1)/4)u[n]$')
axs[2].set_ylim([-1.4,1.4])
fig.tight_layout()
fig.savefig('exp/densum.png')
