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

def plot_convolution(axs, x, x_mset, h, h_mset, y, y_nset, n, y_final):
    '''
    x_mset is the set of m over which x is defined; it must contain successive integers.
    h_mset is the set of m over which h is defined
    y_nset is the set of n over which y is defined
    [n-x_mset] should be a subset of h_mset, and n should be in y_nset
    y should start out all zeros, but should be accumulated over time (as output of this func).
    y_final should be np.convolve(x,h)
    '''
    axs[0].clear()
    axs[0].stem(x_mset,x)
    ylim = [ min(-0.1,1.1*np.amin(x)), max(1.1,1.1*np.amax(x))  ]
    axs[0].plot(x_mset,np.zeros(x_mset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[0].set_title('$x[m]$')
    axs[1].clear()
    hplot = h[np.argwhere(h_mset==n-x_mset[0])[0,0]:(np.argwhere(h_mset==n-x_mset[-1])[0,0]-1):-1]
    axs[1].stem(x_mset, hplot)
    ylim = [ min(-0.1,1.1*np.amin(h)), max(1.1,1.1*np.amax(h))  ]
    axs[1].plot(x_mset,np.zeros(x_mset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[1].set_title('$h[%d-m]$'%(n))
    axs[2].clear()
    y[y_nset==n] = np.sum(hplot*x)
    axs[2].stem(y_nset,y)
    ylim = [ min(-0.1,1.1*np.amin(y_final)), max(1.1,1.1*np.amax(y_final))  ]
    axs[2].plot(y_nset,np.zeros(y_nset.shape),'k-',[0,1e-6],1.1*np.array(ylim),'k-')
    axs[2].set_title('$y[m]=h[m]*x[m]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    return(y)

#############################################################################################
# common axis parameters
N = 64
nset = np.arange(-(N-1),N)
omega = np.linspace(-np.pi,np.pi,N,endpoint=False)
L = 11
omega_ticks = np.pi*np.array([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
omega_ticklabels = ['-π','-3π/4','-π/2','-π/4','0','π/4','π/2','3π/4','π']
n_ticks = np.array([-L, -int((L-1)/2), 0, int((L-1)/2), L])
n_ticklabels = ['-L','-(L-1)/2','0','(L-1)/2','L']

#############################################################################################
# Image showing $l_I[n]$, $L_I(\omega)$, truncated $l[n]$, $L(\omega)$.
M = 9
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
omegac = np.pi/4

hi = (omegac/np.pi)*np.sinc(omegac*nset/np.pi)
HI = np.zeros(len(omega))
HI[np.abs(omega)<=omegac]=1
plotspec(axs[0,1],omega,HI,omega_ticks,omega_ticklabels)
axs[0,1].set_title('$H_{LP,i}(\omega)$, omegac=%s'%(omega_ticklabels[5]))

plotwave(axs[0,0],nset,hi,np.array([-8,-4,0,4,8]),['-2π/ωc','-π/ωc','0','π/ωc','2π/ωc'],16)
axs[0,0].set_title('$h_{LP,i}[n]$, omegac=%s'%(omega_ticklabels[5]))

narr = np.concatenate((np.arange(-2*M,-M+1),np.arange(-M,M+1),np.arange(M,2*M+1)))
h = np.concatenate((np.zeros(-M+1+2*M),hi[(-M<=nset)&(nset<=M)],np.zeros(2*M+1-M)))
H = np.fft.fftshift(np.real(np.fft.fft(np.fft.fftshift(h))))
omeg = np.linspace(-np.pi,np.pi,len(H))
plotspec(axs[1,1],omeg,H,omega_ticks,omega_ticklabels)
axs[1,1].set_title('$H_{LP}(\omega)$, omegac=%s'%(omega_ticklabels[5]))

plotwave(axs[1,0],narr,h,np.array([-8,-4,0,4,8]),['-2π/ωc','-π/ωc','0','π/ωc','2π/ωc'],16)
axs[1,0].set_title('Truncated $h_{LP}[n]$, omegac=%s'%(omega_ticklabels[5]))

fig.tight_layout()
fig.savefig('exp/odd_truncated.png')


#############################################################################################
# rectangles and sincs
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)

plotspec(axs[0,1],omega,HI,omega_ticks,omega_ticklabels)
axs[0,1].set_title('$H_{i}(\omega)$, omegac=%s'%(omega_ticklabels[5]))
plotwave(axs[0,0],nset,hi,np.array([-8,-4,0,4,8]),n_ticklabels,L)
axs[0,0].set_title('$h_{i}[n]$, omegac=%s'%(omega_ticklabels[5]))

wR = np.zeros(len(nset))
wR[np.abs(nset) <= (L-1)/2] = 1
plotwave(axs[1,0],nset,wR,n_ticks, ['-2π/ωc','-π/ωc','0','π/ωc','2π/ωc'],L)
axs[1,0].set_title('$w_R[n]$, length=%d'%(L))
WR = np.zeros(len(omega))
WR[omega==0] = L
WR[omega != 0] = np.sin(omega[omega!=0]*L/2)/np.sin(omega[omega!=0]/2)
plotspec(axs[1,1],omega,WR,omega_ticks,omega_ticklabels)
axs[1,1].set_title('$W_R(\omega)$, length=%d'%(L))

fig.tight_layout()
fig.savefig('exp/rectangles_and_sincs.png')

#############################################################################################
# dirichlet_2pi_period
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

omega2 = np.linspace(-2*np.pi,2*np.pi,2*N,endpoint=False)
omega2_ticks = np.pi*np.array([-2,-1.5,-1,-0.5,0,0.5,1,1.5,2])
omega2_ticklabels = ['-2π','-3π/2','-π','-π/2','0','π/2','π','3π/2','2π']
WR2 = np.zeros(len(omega2))
WR2[omega2==0] = L
WR2[omega2 != 0] = np.sin(omega2[omega2!=0]*L/2)/np.sin(omega2[omega2!=0]/2)
WR2[0]=L
plotspec(ax,omega2,WR2,omega2_ticks,omega2_ticklabels)
ax.set_title('$W_R(\omega)$ is periodic with a period of $2\pi$')
fig.tight_layout()
fig.savefig('exp/dirichlet_2pi_period.png')

#############################################################################################
# dirichlet_dc_is_L
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

plotspec(ax,omega,WR,omega_ticks,omega_ticklabels)
ax.set_yticks(np.array([L]))
ax.set_yticklabels(['L'])
ax.set_title('$W_R(\omega)$ has a peak amplitude of $L$')
fig.tight_layout()
fig.savefig('exp/dirichlet_dc_is_L.png')

#############################################################################################
# dirichlet_and_L_over_omega
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

plotspec(ax,omega,WR,omega_ticks,omega_ticklabels)
ax.plot(omega[omega!=0], 2/np.abs(omega[omega!=0]), 'k--')  # 2L/omega as a dashed line
ax.set_title('$W_R(\omega)$ falls as  $2/\omega$')
fig.tight_layout()
fig.savefig('exp/dirichlet_and_2_over_omega.png')

#############################################################################################
# dirichlet_with_null_frequencies
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

null_ticks = np.pi*np.array([0,2/L,4/L,6/L,8/L])
null_ticklabels = ['0','2π/L','4π/L','6π/L','8π/L']
plotspec(ax,omega[omega>=0],WR[omega>=0],null_ticks,null_ticklabels)
ax.set_title('Frequencies of the null of $W_R(\omega)$')
fig.tight_layout()
fig.savefig('exp/dirichlet_with_null_frequencies.png')

#############################################################################################
# dirichlet_with_peak_frequencies
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

peak_ticks = np.pi*np.array([0,3/L,5/L,7/L])
peak_ticklabels = ['0','3π/L','5π/L','7π/L']
plotspec(ax,omega[omega>=0],WR[omega>=0],peak_ticks,peak_ticklabels)
ax.set_title('Frequencies of the peaks of $W_R(\omega)$')
fig.tight_layout()
fig.savefig('exp/dirichlet_with_peak_frequencies.png')

#############################################################################################
# dirichlet_in_decibels
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

levels = 20*np.log10(np.maximum(0.001, np.abs(WR[omega>=0]/WR[omega==0])))
plotspec(ax,omega[omega>=0],levels,peak_ticks,peak_ticklabels)
ax.set_title('$W_R(\omega)$ in decibels')
fig.tight_layout()
fig.savefig('exp/dirichlet_in_decibels.png')

#############################################################################################
# bartlettwindow
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

wB = np.maximum(0, 1-2*np.abs(nset)/(L-1))
plotwave(ax, nset, wB, n_ticks, n_ticklabels, L)
fig.tight_layout()
fig.savefig('exp/bartlettwindow.png')

#############################################################################################
# rect_convolve
# Video showing the convolution of a small rectangle with a small rectangle to produce a triangle
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
x_mset = np.arange(-L,L+1,dtype='int')
x = np.zeros(len(x_mset))
x[np.abs(x_mset) <= (L-3)/4] = np.sqrt(2/(L-1))
h_mset = np.arange(-2*L-1,2*L+2,dtype='int')
h = np.zeros(len(h_mset))
h[np.abs(h_mset) <= (L-3)/4] = np.sqrt(2/(L-1))
y_nset = x_mset
y = np.zeros(len(y_nset))
y_final = np.convolve(x,h)
for n in y_nset:
    y  = plot_convolution(axs, x, x_mset, h, h_mset, y, y_nset, n, y_final)
    fig.tight_layout()
    fig.savefig('exp/weightedaverage%d.png'%(n+L))

#############################################################################################
# bartlett_small_rectangle_and_spectrum

fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(2,1)

wsmall = np.zeros(len(nset))
wsmall[np.abs(nset) <= (L-3)/4] = np.sqrt(2/(L-1))
plotwave(axs[0],nset,wsmall, n_ticks, n_ticklabels, L)
axs[0].set_title('Rectangle with length (L-1)/2, height sqrt(2/(L-1))')

b_ticks = np.pi*np.array([0,4/(L-1),8/(L-1)])
b_ticklabels = ['0','4π/(L-1)','8π/(L-1)']

Wsmall = np.zeros(len(omega))
Wsmall[omega==0] = np.sqrt((L-1)/2)
Wsmall[omega != 0] = np.sqrt(2/(L-1))*np.sin(omega[omega!=0]*(L-1)/4)/np.sin(omega[omega!=0]/2)
plotspec(axs[1],omega[omega>=0],Wsmall[omega>=0],b_ticks,b_ticklabels)
axs[1].set_title('DTFT of the small rectangle')

fig.tight_layout()
fig.savefig('exp/bartlett_small_rectangle_and_spectrum.png')

#############################################################################################
# bartlett_and_rectangle_spectrum

fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(2,1)

plotwave(axs[0],nset,wB, n_ticks, n_ticklabels, L)
axs[0].set_title('Bartlett window w/length L')

plotspec(axs[1],omega,Wsmall**2,b_ticks,b_ticklabels)
axs[1].set_title('DTFT of Bartlett window')

fig.tight_layout()
fig.savefig('exp/bartlett_and_rectangle_spectrum.png')

#############################################################################################
# bartlett_in_decibels

fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(2,1)

plotwave(axs[0],nset,wB, n_ticks, n_ticklabels, L)
axs[0].set_title('Bartlett window w/length L')

Wlevel = 20*np.log10(np.maximum(0.001,Wsmall**2/Wsmall[omega==0]**2))
plotspec(axs[1],omega,Wlevel,b_ticks,b_ticklabels)
axs[1].set_title('DTFT of Bartlett window, in decibels')

fig.tight_layout()
fig.savefig('exp/bartlett_in_decibels.png')

#############################################################################################
# Hann window
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

wN = np.zeros(len(nset))
wN[np.abs(nset) <= (L-1)/2] = np.hanning(L)
plotwave(ax, nset, wN, n_ticks, n_ticklabels, L)
ax.plot(np.arange(-int((L-1)/2),int((L+1)/2)), np.hanning(L))
fig.tight_layout()
fig.savefig('exp/hannwindow.png')

#############################################################################################
# Hann window: first piece
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

plotspec(ax,omega,0.5*WR,omega_ticks,omega_ticklabels)
ax.set_title('Center term of Hann spectrum is just $0.5W_R(\omega)$')
fig.tight_layout()
fig.savefig('exp/hann1piece.png')

#############################################################################################
# Hann window: second piece
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

WN1 = np.zeros(len(omega))
WN1[omega==0] = L
WN1[omega != 0] = np.sin(omega[omega!=0]*L/2)/np.sin(omega[omega!=0]/2)
omegashift = omega - 2*np.pi/(L-1)
WN2 = np.zeros(len(omegashift))
WN2[omegashift==0] = L
WN2[omegashift != 0] = np.sin(omegashift[omegashift!=0]*L/2)/np.sin(omegashift[omegashift!=0]/2)

plotspec(ax,omega,0.5*WN1+0.25*WN2,omega_ticks,omega_ticklabels)
ax.plot(omega,0.5*WN1,'k--')
ax.plot(omega,0.25*WN2,'k--')
ax.set_ylim(1.1*np.amin(np.concatenate((0.5*WN1,0.25*WN2,0.5*WN1+0.25*WN2))),1.1*np.amax(np.concatenate((0.5*WN1,0.25*WN2,0.5*WN1+0.25*WN2))))
ax.set_title('First two terms are $0.5W_R(\omega)+0.25W_R(\omega-2\pi/(L-1))$')
fig.tight_layout()
fig.savefig('exp/hann2piece.png')

#############################################################################################
# Hann window: third piece
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

omegashift = omega + 2*np.pi/(L-1)
WN3 = np.zeros(len(omegashift))
WN3[omegashift==0] = L
WN3[omegashift != 0] = np.sin(omegashift[omegashift!=0]*L/2)/np.sin(omegashift[omegashift!=0]/2)

plotspec(ax,omega,0.5*WN1+0.25*WN2+0.25*WN3,omega_ticks,omega_ticklabels)
ax.plot(omega,0.5*WN1,'k--')
ax.plot(omega,0.25*WN2,'k--')
ax.plot(omega,0.25*WN3,'k--')
ax.set_ylim(1.1*np.amin(np.concatenate((0.5*WN1,0.25*WN2,0.25*WN3,0.5*WN1+0.25*WN2+0.25*WN3))),1.1*np.amax(np.concatenate((0.5*WN1,0.25*WN2,0.25*WN3,0.5*WN1+0.25*WN2+0.25*WN3))))
ax.set_title('All 3 terms of the Hann window spectrum')
fig.tight_layout()
fig.savefig('exp/hann3piece.png')

#############################################################################################
# Hamming window
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

wM = np.zeros(len(nset))
wM[np.abs(nset) <= (L-1)/2] = np.hamming(L)
plotwave(ax, nset, wM, n_ticks, n_ticklabels, L)
ax.plot(np.arange(-int((L-1)/2),int((L+1)/2)), np.hamming(L))
fig.tight_layout()
fig.savefig('exp/hammingwindow.png')

#############################################################################################
# Hamming 3 piece
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()

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

plotspec(ax,omega,0.54*WM1+0.23*WM2+0.23*WM3,omega_ticks,omega_ticklabels)
ax.plot(omega,0.54*WM1,'k--')
ax.plot(omega,0.23*WM2,'k--')
ax.plot(omega,0.23*WM3,'k--')
ax.set_ylim(1.1*np.amin(np.concatenate((0.54*WM1,0.23*WM2,0.23*WM3,0.54*WM1+0.23*WM2+0.23*WM3))),1.1*np.amax(np.concatenate((0.54*WM1,0.23*WM2,0.23*WM3,0.54*WM1+0.23*WM2+0.23*WM3))))
fig.tight_layout()
fig.savefig('exp/hamming3piece.png')
