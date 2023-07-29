import numpy  as np
import matplotlib.figure, subprocess, os

os.makedirs('exp', exist_ok=True)
            
def plotspec(ax,omega,X,xticks,xticklabels):
    ax.plot(omega,np.zeros(len(omega)),'k-') # omega axis
    ax.plot([0,1e-6],[np.amin(X)-0.1,np.amax(X)+0.1],'k-') # X axis
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(np.amin(omega),np.amax(omega))
    ax.set_ylim(np.amin(X)-0.1,np.amax(X)+0.1)
    ax.plot(omega,X,'b-')

def plotwave(ax,nset,x,xticks,xticklabels,L):
    ax.stem(nset,x)
    ax.plot(nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
    ax.set_xlim(-L,L)
    ax.set_ylim(np.amin(x)-0.1,np.amax(x)+0.1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

def plot_convolution(axs, x, x_mset, h, h_mset, y, y_nset, n, y_final, domega, xticks, xticklabels):
    '''
    Frequency domain convolution
    x_mset is the set of m over which x is defined; it must contain successive integers.
    h_mset is the set of m over which h is defined
    y_nset is the set of n over which y is defined
    [n-x_mset] should be a subset of h_mset, and n should be in y_nset
    y should start out all zeros, but should be accumulated over time (as output of this func).
    y_final should be np.convolve(x,h)
    x_mset is multiplied by domega
    xticks specifies where to label the X-axis
    xticklabels specifies what to label it
    '''
    axs[0].clear()
    axs[0].plot(x_mset*domega,x)
    ylim = [ min(-0.1,1.1*np.amin(x)), max(1.1,1.1*np.amax(x))  ]
    axs[0].plot(x_mset*domega,np.zeros(x_mset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[0].set_title('$H_i(θ)$')
    axs[1].clear()
    hplot = h[np.argwhere(h_mset==n-x_mset[0])[0,0]:(np.argwhere(h_mset==n-x_mset[-1])[0,0]-1):-1]
    axs[1].plot(x_mset*domega, hplot)
    ylim = [ min(-0.1,1.1*np.amin(h)), max(1.1,1.1*np.amax(h))  ]
    axs[1].plot(x_mset*domega,np.zeros(x_mset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[1].set_title('$W(%2.2g-θ)$'%(n*domega))
    axs[2].clear()
    y[y_nset==n] = np.sum(hplot*x)
    axs[2].plot(y_nset*domega,y)
    ylim = [ min(-0.1,1.1*np.amin(y_final)), max(1.1,1.1*np.amax(y_final))  ]
    axs[2].plot(y_nset*domega,np.zeros(y_nset.shape),'k-',[0,1e-6],1.1*np.array(ylim),'k-')
    axs[2].set_title('$H(θ)=H_i(θ)*W(θ)$')
    axs[2].set_xlabel('$θ$')
    fig.tight_layout()
    return(y)

#############################################################################################
# common axis parameters
N = 64
nset = np.arange(-(N-1),N,dtype='int')
domega = 2*np.pi/(2*N+1)
omega = nset * domega
omega_ticks = np.pi*np.array([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
omega_ticklabels = ['-π','-3π/4','-π/2','-π/4','0','π/4','π/2','3π/4','π']

#############################################################################################
# Specify the ideal filter

omegac = np.pi/4
HI = np.zeros(len(omega))
HI[np.abs(omega)<=omegac]=1

#############################################################################################
# rectangles and sincs

L = 11
WR = np.zeros(len(omega))
WR[omega==0] = L
WR[omega != 0] = np.sin(omega[omega!=0]*L/2)/np.sin(omega[omega!=0]/2)

#############################################################################################
# freq_convolve
# Video showing the convolution of an ideal filter with the transform of a rectangle window
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
x = HI
x_mset = nset
h = np.concatenate((WR,WR,WR))
h_mset = np.arange(-(2*N+1)-(N-1),N+(2*N+1),dtype='int')
y = np.zeros(len(nset))
y_nset = x_mset
y_final = np.convolve(x,h)
for n in y_nset:
    y  = plot_convolution(axs, x, x_mset, h, h_mset, y, y_nset, n, y_final, domega, omega_ticks, omega_ticklabels)
    fig.tight_layout()
    fig.savefig('exp/ideal_with_rect%d.png'%(n-min(y_nset)))

#############################################################################################
# Hamming window

WM1 = np.zeros(len(omega))
WM1[omega==0] = L
WM1[omega != 0] = np.sin(omega[omega!=0]*L/2)/np.sin(omega[omega!=0]/2)
omegashift = omega - 2*np.pi/(L-1)
WM2 = np.zeros(len(omegashift))
WM2[omegashift==0] = L
WM2[omegashift != 0] = np.sin(omegashift[omegashift!=0]*L/2)/np.sin(omegashift[omegashift!=0]/2)
omegashift = omega + 2*np.pi/(L-1)
WM3 = np.zeros(len(omegashift))
WM3[omegashift==0] = L
WM3[omegashift != 0] = np.sin(omegashift[omegashift!=0]*L/2)/np.sin(omegashift[omegashift!=0]/2)
WM = 0.54*WM1 + 0.23*WM2 + 0.23*WM3

#############################################################################################
# freq_convolve
# Video showing the convolution of an ideal filter with the transform of a Hamming window
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
x = HI
x_mset = nset
h = np.concatenate((WM,WM,WM))
h_mset = np.arange(-(2*N+1)-(N-1),N+(2*N+1),dtype='int')
y = np.zeros(len(nset))
y_nset = x_mset
y_final = np.convolve(x,h)
for n in y_nset:
    y  = plot_convolution(axs, x, x_mset, h, h_mset, y, y_nset, n, y_final, domega, omega_ticks, omega_ticklabels)
    fig.tight_layout()
    fig.savefig('exp/ideal_with_hamming%d.png'%(n-min(y_nset)))
