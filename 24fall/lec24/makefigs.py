import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.makedirs('exp',exist_ok=True)

##########################################################################################3
# delaycircular49.png
#  video of an impulse getting more and more delayed until it wraps
#  around (left), while (right) the sampled magnitude spectrum stays
#  the same, but the sampled phase spectrum gets larger and larger tilt

N = 32

fig = plt.figure(figsize=(10,4))
gs = GridSpec(2,2,figure=fig)
ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,1])

for n0 in range(50):
    nset = np.arange(0,N)
    x = np.zeros(len(nset))
    x[nset== (n0 % N)] = 1
    ax0.clear()
    ax0.stem(nset, x)
    ax0.set_xlim([-N,2*N])
    ax0.set_xticks([0,N])
    ax0.set_xticklabels(['0','N'])
    ax0.set_xlabel('$n$')
    ax0.set_title('$\delta[((n-%d))_{%d}]$'%(n0,N))

    kset = np.arange(0,N)
    Xmag = np.ones(len(kset))
    ax1.clear()
    ax1.stem(kset, Xmag)
    ax1.set_title('$|X[k]|=1$')

    Xang = -2*np.pi*kset*n0/N
    ax2.clear()
    ax2.stem(kset, Xang)
    ax2.set_ylim([-2*np.pi*max(kset)*50/N, 0])
    ax2.set_xlabel('$k$')
    ax2.set_title('$∠X[k]=-2πkn_0/N$')

    fig.tight_layout()
    fig.savefig('exp/delaycircular%d.png'%(n0))

###########################################################################################3
# delayperiodic49.png
#
#  video of the periodic-in-time version of the same impulse being
#  delayed more and more (left), while (right) is the same
#

N = 32

fig = plt.figure(figsize=(10,4))
gs = GridSpec(2,2,figure=fig)
ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,1])

for n0 in range(50):
    nset = np.arange(-N,2*N)
    x = np.zeros(len(nset))
    x[nset== (n0 % N)-N] = 1
    x[nset== (n0 % N)] = 1
    x[nset== (n0 % N)+N] = 1
    ax0.clear()
    ax0.stem(nset, x)
    ax0.set_xlim([-N,2*N])
    ax0.set_xlabel('$n$')
    ax0.set_xticks([-N,0,N,2*N])
    ax0.set_xticklabels(['-N','0','N','2N'])
    ax0.set_title('$\delta[((n-%d))_{%d}]$'%(n0,N))

    kset = np.arange(0,N)
    Xmag = np.ones(len(kset))
    ax1.clear()
    ax1.stem(kset, Xmag)
    ax1.set_title('$|X[k]|=1$')

    Xang = -2*np.pi*kset*n0/N
    ax2.clear()
    ax2.stem(kset, Xang)
    ax2.set_ylim([-2*np.pi*max(kset)*50/N, 0])
    ax2.set_xlabel('$k$')
    ax2.set_title('$∠X[k]=-2πkn_0/N$')

    fig.tight_layout()
    fig.savefig('exp/delayperiodic%d.png'%(n0))

##################################################################################################
# principalphase49.png
#
#  video of the periodic-in-time version of the same impulse being
#  delayed more and more (left), while (right) the sampled unwrapped
#  phase keeps getting bigger, but the sampled principal-phase spectrum
#  wraps around more and more
#  

N = 32

fig = plt.figure(figsize=(10,4))
gs = GridSpec(2,2,figure=fig)
ax0 = fig.add_subplot(gs[:,0])
#ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])

for n0 in range(50):
    nset = np.arange(-N,2*N)
    x = np.zeros(len(nset))
    x[nset== (n0 % N)-N] = 1
    x[nset== (n0 % N)] = 1
    x[nset== (n0 % N)+N] = 1
    ax0.clear()
    ax0.stem(nset, x)
    ax0.set_xlim([-N,2*N])
    ax0.set_xticks([-N,0,N,2*N])
    ax0.set_xticklabels(['-N','0','N','2N'])
    ax0.set_xlabel('$n$')
    ax0.set_title('$\delta[((n-%d))_{%d}]$'%(n0,N))

    kset = np.arange(0,N)
    Xmag = np.ones(len(kset))
    #ax1.clear()
    #ax1.stem(kset, Xmag)
    #ax1.set_title('$|X[k]|=1$')

    Xang = -2*np.pi*kset*n0/N
    ax2.clear()
    ax2.stem(kset, Xang)
    ax2.set_ylim([-2*np.pi*max(kset)*50/N, 0])
    ax2.set_xlabel('$k$')
    ax2.set_title('Unwrapped Phase')

    ax3.clear()
    ax3.stem(kset, -np.pi + ((Xang+np.pi)%(2*np.pi)))
    ax3.set_ylim([-np.pi, np.pi])
    ax3.set_xlabel('$k$')
    ax3.set_title('Principal Phase')
    
    fig.tight_layout()
    fig.savefig('exp/principalphase%d.png'%(n0))

#############################################################################################
# convolutionlinear.png
#
#  Image showing linear convolution of two squares, and product of their two DFTs

N = 32
L = 20

fig = plt.figure(figsize=(6,4))
axs = fig.subplots(3,2)

nset = np.arange(-N,2*N)
h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = 1
axs[0,0].stem(nset, h)
axs[0,0].set_title('$h[n]$')
axs[1,0].stem(nset, h)
axs[1,0].set_title('$x[n]$')
y = np.zeros(len(nset))
y[(0 <= nset) & (nset < 2*L-1)] = np.convolve(np.ones(L), np.ones(L))
axs[2,0].stem(nset, y)
axs[2,0].set_title('$y[n]=x[n]*h[n]')
axs[2,0].set_xlabel('$n$')

kset = np.arange(0,N)
Hmag = np.abs(np.fft.fft(h[N:2*N]))
axs[0,1].stem(kset,Hmag)
axs[0,1].set_title('$|H[k]|$')
axs[1,1].stem(kset,Hmag)
axs[1,1].set_title('$|X[k]|$')
axs[2,1].stem(kset,Hmag*Hmag)
axs[2,1].set_title('$|Y[k]|=|H[k]X[k]|$')
axs[2,1].set_xlabel('$k$')

fig.tight_layout()
fig.savefig('exp/convolutionlinear.png')

#############################################################################################
# convolutionperiodic.png
#
#   Image showing circular convolution of two periodic-in-time squares, and product of their two DFTs

N = 32
L = 20

fig = plt.figure(figsize=(6,4))
axs = fig.subplots(3,2)

nset = np.arange(-N,2*N)
h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = 1
h[(-N <= nset) & (nset < -N+L)] = 1
h[(N <= nset) & (nset < N+L)] = 1
axs[0,0].stem(nset, h)
axs[0,0].set_title('Periodic $h[n]$')
axs[1,0].stem(nset, h)
axs[1,0].set_title('Periodic $x[n]$')
y = np.real(np.fft.ifft(np.fft.fft(h)**2))
axs[2,0].stem(nset, y)
axs[2,0].set_title('Periodic $y[n]=x[n]*h[n]$')
axs[2,0].set_xlabel('$n$')

kset = np.arange(0,N)
Hmag = np.abs(np.fft.fft(h[N:2*N]))
axs[0,1].stem(kset,Hmag)
axs[0,1].set_title('$|H[k]|$')
axs[1,1].stem(kset,Hmag)
axs[1,1].set_title('$|X[k]|$')
axs[2,1].stem(kset,Hmag*Hmag)
axs[2,1].set_title('$|Y[k]|=|H[k]X[k]|$')
axs[2,1].set_xlabel('$k$')

fig.tight_layout()
fig.savefig('exp/convolutionperiodic.png')

###############################################################################################
# circularsquares49.png
#
# Video showing circular convolution of a square with a pair of impulses

N = 50
L = 15
M = 45

fig = plt.figure(figsize=(8,4))
axs = fig.subplots(3,2)

nset = np.arange(N)
x = np.zeros(len(nset))
x[int(M/3)] = 1
x[M-1] = 1
axs[0,0].stem(nset, x)
axs[0,0].set_title('$x[n]$')
h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = 1
axs[1,0].stem(nset, h)
axs[1,0].set_title('$h[n]$')
y = np.real(np.fft.ifft(np.fft.fft(h)*np.fft.fft(x)))
axs[2,0].stem(nset, y)
axs[2,0].set_title('$y[n]=x[n]*h[n]$ circular')
axs[2,0].set_xlabel('$n$')

mset = nset[:]
axs[0,1].stem(mset, x)
axs[0,1].set_title('$x[m]$')

ybuild = np.zeros(len(nset))
for n in range(N):
    hbuild = h[(n-mset)%N]
    axs[1,1].clear()
    axs[1,1].stem(nset,hbuild)
    axs[1,1].set_title('$h[((%d-m))_{%d}]$'%(n,N))
    ybuild[n] = np.sum(hbuild * x)
    axs[2,1].clear()
    axs[2,1].stem(mset, ybuild)
    axs[2,1].set_title('$y[m]=x[m]*h[m]$ circular')
    axs[2,1].set_xlabel('$m$')
    axs[2,1].set_ylim([0,1.1*np.amax(y)])
    fig.tight_layout()
    fig.savefig('exp/circularsquares%d.png'%(n))

###############################################################################################
# circularexps49.png
#
# Video showing circular convolution of an exponential with a pair of impulses

N = 50
L = 25
M = 45

fig = plt.figure(figsize=(8,4))
axs = fig.subplots(3,2)

nset = np.arange(N)
x = np.zeros(len(nset))
x[int(M/3)] = 1
x[M-1] = 1
axs[0,0].stem(nset, x)
axs[0,0].set_title('$x[n]$')
h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = np.exp(-np.arange(L)*3/L)
axs[1,0].stem(nset, h)
axs[1,0].set_title('$h[n]$')
y = np.real(np.fft.ifft(np.fft.fft(h)*np.fft.fft(x)))
axs[2,0].stem(nset, y)
axs[2,0].set_title('$y[n]=x[n]*h[n]$ circular')
axs[2,0].set_xlabel('$n$')

mset = nset[:]
axs[0,1].stem(mset, x)
axs[0,1].set_title('$x[m]$')

ybuild = np.zeros(len(nset))
for n in range(N):
    hbuild = h[(n-mset)%N]
    axs[1,1].clear()
    axs[1,1].stem(nset,hbuild)
    axs[1,1].set_title('$h[((%d-m))_{%d}]$'%(n,N))
    ybuild[n] = np.sum(hbuild * x)
    axs[2,1].clear()
    axs[2,1].stem(mset, ybuild)
    axs[2,1].set_title('$y[m]=x[m]*h[m]$ circular')
    axs[2,1].set_xlabel('$m$')
    axs[2,1].set_ylim([0,1.1*np.amax(y)])
    fig.tight_layout()
    fig.savefig('exp/circularexps%d.png'%(n))

###############################################################################################
# convolutionlengths.png
#
#   Show an image with h[n] labeled L-1, x[n] labeled M-1, y[n] labeled L+M-2

L = 25
M = 45
N = L+M-1

fig = plt.figure(figsize=(8,4))
axs = fig.subplots(3,1)

nset = np.arange(N)
x = np.zeros(len(nset))
x[:M] = 1
axs[0].stem(nset, x)
axs[0].set_title('$x[n]$')
axs[0].set_xticks([0,M-1])
axs[0].set_xticklabels(['0','M-1'])

h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = np.exp(-np.arange(L)*2/L)
axs[1].stem(nset, h)
axs[1].set_title('$h[n]$')
axs[1].set_xticks([0,L-1])
axs[1].set_xticklabels(['0','L-1'])

y = np.real(np.fft.ifft(np.fft.fft(h)*np.fft.fft(x)))
axs[2].stem(nset, y)
axs[2].set_title('$y[n]=x[n]*h[n]$ circular')
axs[2].set_xlabel('$n$')
axs[2].set_xticks([0,N-1])
axs[2].set_xticklabels(['0','N-1=L+M-2'])

fig.tight_layout()
fig.savefig('exp/convolutionlengths.png')

###############################################################################################3
# circandlinear74.png
#
#  Left: images of x, h, and x*h with N=M
#
#  Right: images of x, h, and x*h with N=L+M-1
#

L = 25
M = 45
N = L+M-1

fig = plt.figure(figsize=(8,4))
axs = fig.subplots(3,2)

nset = np.arange(N)
x = np.zeros(len(nset))
x[:M] = 1
axs[0,0].stem(nset, x)
axs[0,0].set_title('$x[n]$')
h = np.zeros(len(nset))
h[(0 <= nset) & (nset < L)] = np.exp(-np.arange(L)*2/L)
axs[1,0].stem(nset, h)
axs[1,0].set_title('$h[n]$')
y = np.real(np.fft.ifft(np.fft.fft(h)*np.fft.fft(x)))
axs[2,0].stem(nset, y)
axs[2,0].set_title('$y[n]=x[n]*h[n]$ circular')
axs[2,0].set_xlabel('$n$')

mset = nset[:]
axs[0,1].stem(mset, x)
axs[0,1].set_title('$x[m]$')

ybuild = np.zeros(len(nset))
for n in range(N):
    hbuild = h[(n-mset)%N]
    axs[1,1].clear()
    axs[1,1].stem(nset,hbuild)
    axs[1,1].set_title('$h[((%d-m))_{%d}]$'%(n,N))
    ybuild[n] = np.sum(hbuild * x)
    axs[2,1].clear()
    axs[2,1].stem(mset, ybuild)
    axs[2,1].set_title('$y[m]=x[m]*h[m]$ circular')
    axs[2,1].set_xlabel('$m$')
    axs[2,1].set_ylim([0,1.1*np.amax(y)])
    fig.tight_layout()
    fig.savefig('exp/circandlinear%d.png'%(n))
