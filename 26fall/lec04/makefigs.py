import numpy as np
import math,subprocess,os
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

os.makedirs('exp',exist_ok=True)

################################################################################
# Probably globally useful stuff
zeromarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='none')
polemarker = matplotlib.markers.MarkerStyle(marker='x',fillstyle='none')
fillmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='full')
omega = np.linspace(0,np.pi,5000)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
ucx = np.cos(2*omega)
ucy = np.sin(2*omega)
def plot_circle(ax, pole=None, ptext='a'):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    if pole != None:
        for k in range(len(pole)):
            s = np.sign(np.imag(pole[k]))
            ax.scatter(x=np.real(pole[k]),y=np.imag(pole[k]),s=40,c='b',marker=polemarker)
            ax.text(x=np.real(pole[k])-0.1,y=np.imag(pole[k])-0.2*s-0.05,s=ptext[k])
    ax.set_aspect('equal')

def plot_spec(ax, omega, H):
    ax.plot(omega,np.zeros(len(omega)),'k-')
    ax.plot(omega,np.abs(H))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Frequency ($\omega$)')
    ax.set_title('$|H(\omega)|$')

def plot_convolution(axs, x, h, y, nset, n, yylim):
    '''
    h should be defined with the same resolution as nset, but not the same range:
    its range should be from 0 to max(nset)-min(nset).
    y should start out all zeros, but should be accumulated over time (as output of this func).
    Actually, if nset not integers, stem will give weird response.
    '''
    axs[0].stem(nset,x)
    ylim = [ min(-0.1,1.1*np.amin(x)), max(1.1,1.1*np.amax(x))  ]
    axs[0].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[0].set_title('$x[m]=\delta[m]$')
    axs[1].clear()
    hplot = np.zeros(nset.shape)
    hplot[nset <= n] = h[n-nset[nset<=n]]
    axs[1].stem(nset,hplot)
    ylim = [ min(-1.1,1.1*np.amin(h)), max(1.1,1.1*np.amax(h))  ]
    axs[1].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[1].set_title('$h[%d-m]$'%(n))
    axs[2].clear()
    y[nset==n] = np.sum(hplot[nset%1==0]*x[nset%1==0])
    axs[2].stem(nset,y)
    axs[2].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],1.1*np.array(yylim),'k-')
    axs[2].set_title('$y[m]=h[m]*x[m]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    return(y)

###########################################################################
# Video showing the convolution of random input with 7-tap weighted averager
# unbounded output
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
nset = np.arange(-int(0.2*N),N+1)
x = np.random.randn(len(nset))
h = np.zeros(len(nset))
h[nset==0] = 0.5
h[np.abs(nset)==1] = 0.25
h[np.abs(nset)==2] = 0.15
h[np.abs(nset)==3] = 0.1
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/weightedaverage%d.png'%(n))

###########################################################################
# Video showing the convolution of step function with step function, generating
# unbounded output
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
nset = np.arange(-int(0.2*N),N+1)
x = np.concatenate((np.zeros(int(0.2*N)), np.ones(N+1)))
h = np.ones(len(nset))
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/stepfunction%d.png'%(n))

###########################################################################
# Video showing the convolution of delta function with exponential, generating
# unbounded output
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
nset = np.arange(-int(0.2*N),N+1)
x = np.zeros(len(nset))
x[nset==0] = 1
h = np.exp(np.log(1.1)*np.arange(len(nset)))
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [0.5*np.amin(yy), 0.5*np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/exponentialresponse%d.png'%(n))
###########################################################################
# Video showing the convolution of delta function with exponential, generating
# unbounded output
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
nset = np.arange(-int(0.2*N),N+1)
x = np.concatenate((np.zeros(int(0.2*N)), np.ones(N+1)))
h = np.exp(np.log(1.1)*np.arange(len(nset)))
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [0.1*np.amin(yy), 0.1*np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/exponentialstepresponse%d.png'%(n))
