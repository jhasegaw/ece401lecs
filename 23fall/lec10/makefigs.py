import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure, subprocess, os, wave

os.makedirs('exp',exist_ok=True)

################################################################################
# Probably globally useful stuff
def plot_spectrum(ax, X, tupslabel, discrete):
    '''
    Plot axis dashes from -2.5pi to 2.5pi, and from 0 to 1.25*max(abs(X)).
    Add boundaries at -2pi, -pi, pi, and 2pi, and elipses at +/- 2.5pi.
    Stem abs(X) at k*omega0.
    Mark xticklabels at xticks.
    For t in tups, enter text t[0] at (t[1],t[2])

    @param:
    ax - axes
    X - coefficients of CT signal
    tupslabel - 'X' or 'Y'
    discrete - True or False
    '''
    N = len(X)
    M = int((N-1)/2)
    omega0 = np.pi/(M+1)
    k = np.arange(-M,M+1)
    xmax = 1.5*np.amax(np.abs(X))
    fmax = 2.5*np.pi
    ax.stem(omega0*k,np.abs(X))
    if discrete:
        lo_f = omega0*k - 2*np.pi
        ax.stem(lo_f[lo_f > -fmax],np.abs(X[lo_f > -fmax]))
        hi_f = omega0*k + 2*np.pi
        ax.stem(hi_f[hi_f < fmax],np.abs(X[hi_f < fmax]))
        
    ax.plot(-fmax,fmax,[0,0],'k-',[0,1e-6],[0,xmax],'k-')

    boundaries = np.array([-2,-1,1,2])*np.pi
    for boundary in boundaries:
        ax.plot([boundary,boundary+1e-6],[0,xmax],'k--')
    ax.annotate(text='...',xy=(-fmax,0.2*xmax),fontsize=18)
    ax.annotate(text='...',xy=(fmax,0.2*xmax),fontsize=18)

    xticks = np.array([-2,-1,-4/(M+1),-2/(M+1),0,2/(M+1),4/(M+1),1,2])*np.pi
    ax.set_xticks(xticks)
    if discrete:
        xticklabels=['$-2\pi$','$-\pi$','$-4\omega_0$','$-2\omega_0$','$0$',
                     '$2\omega_0$','$4\omega_0$','$\pi$','$2\pi$']
    else:
        xticklabels=['$-F_s$','$-F_s/2$','$-4F_0$','$-2F_0$','$0$',
                     '$2F_0$','$4F_0$','$F_s/2$','$F_s$']
    ax.set_xticklabels(xticklabels,fontsize=18)
    tups = [ ('$%s_%d$'%(tupslabel,i),omega0*k[M+i],np.abs(X[M+i])) for i in range(M+1) ]    
    tups += [ ('$%s_%d^*$'%(tupslabel,i),omega0*k[M-i],np.abs(X[M+i])) for i in range(1,M+1) ]    
    for t in tups:
        ax.annotate(text=t[0],xy=(t[1],t[2]),fontsize=18)
    ax.set_xlim(-fmax,fmax)
    ax.set_ylim(0,xmax)

def plot_convolution(ax, n_lo, n_hi, x_lo, x_hi, h, H0, filename):
    '''
    Top axes: 
    plot(n_hi,x_hi)
    plot axes over n_hi, and from min(x_hi)-0.1 to max(x_hi)+0.1,
    on both top and bottom axes.
    create images with each valid h,
    each one creating a sample of y on the second axis (divided by H0).
    Save them to filename+'%d.png'%(n-len(h)+1).
    '''
    xmin = np.amin(x_hi)-0.1
    xmax = np.amax(x_hi)+0.1
    ax[0].plot(n_hi,np.zeros(len(n_hi)),'k-',[0,1e-6],[xmin,xmax],'k-')
    ax[1].plot(n_hi,np.zeros(len(n_hi)),'k-',[0,1e-6],[xmin,xmax],'k-')
    for n in range(len(n_lo)-len(h)):
        ax[0].clear()
        ax[0].plot(n_hi,np.zeros(len(n_hi)),'k-',[0,1e-6],[xmin,xmax],'k-')
        ax[0].plot(n_hi,x_hi)
        xh = x_lo[n:n+len(h)] * np.flip(h)
        ax[0].stem(n_lo[n:n+len(h)], xh)
        ax[0].set_title('Product of filter times input, $h[m]x[n-m]$',fontsize=18)
        ax[1].stem([n+int(len(h)/2)],np.sum(xh)/H0)
        ax[1].set_title('Output signal $y[n]=\sum_m h[m]x[n-m]$',fontsize=18)
        fig.tight_layout()
        fig.savefig(filename+'%d.png'%(n))
        
###########################################################################
# Spectrum of x(t)
fig, ax = plt.subplots(1,1,figsize=(14,3))
X = 1-0.5*np.abs(np.arange(-5,6))/6
plot_spectrum(ax, X, 'X', False)
ax.set_title('Spectrum of $x(t)$',fontsize=18)
ax.set_xlabel('Frequency (Hz)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/spectrum_xt.png')

fig, ax = plt.subplots(1,1,figsize=(14,3))
plot_spectrum(ax, X, 'X', True)
ax.set_title('Spectrum of $x[n]$',fontsize=18)
ax.set_xlabel('Frequency (radians/sample)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/spectrum_xn.png')

Y = 0.5+0.5*np.abs(np.arange(-5,6))/6
fig, ax = plt.subplots(1,1,figsize=(14,3))
plot_spectrum(ax, Y, 'Y', True)
ax.set_title('Spectrum of $y[n]$',fontsize=18)
ax.set_xlabel('Frequency (radians/sample)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/spectrum_yn.png')

fig, ax = plt.subplots(1,1,figsize=(14,3))
plot_spectrum(ax, Y, 'Y', False)
ax.set_title('Spectrum of $y(t)$',fontsize=18)
ax.set_xlabel('Frequency (Hz)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/spectrum_yt.png')

########################################################################3
# Xk and Yk, then xt and yt, rectangular averager
fig, ax = plt.subplots(2,1,figsize=(14,8))
X = 1-0.5*np.abs(np.arange(-5,6))/6
plot_spectrum(ax[0], X, 'X', True)

N = len(X)
M = int((N-1)/2)
L = 5
omega = np.arange(-M,M+1)*np.pi/(M+1)
H = np.ones(N)
for i,w in enumerate(omega):
    if np.abs(w)>0:
        H[i] = np.sin(L*w/2)/(L*np.sin(w/2))
plot_spectrum(ax[1], H*X, 'Y', True)
ax[0].set_title('Spectrum of $x[n]$',fontsize=18)
ax[1].set_title('Spectrum of $y[n]=h[n]\\ast x[n]$',fontsize=18)
ax[1].set_xlabel('Frequency (radians/sample)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/averager_spectra.png')

fig, ax = plt.subplots(2,1,figsize=(14,8))
n_axis = np.linspace(0,36,300)
x = np.zeros(len(n_axis))
y = np.zeros(len(n_axis))
for k,w in enumerate(omega):
    x += np.real(X[k]*np.exp(1j*w*n_axis))
    y += np.real(H[k]*X[k]*np.exp(1j*w*n_axis))
ax[0].plot(n_axis,x)
ax[1].plot(n_axis,y)
ax[0].set_title('Input Waveform $x(t)$',fontsize=18)
ax[1].set_title('Output Waveform $y(t)$, Averaged with a 5-sample Averager',fontsize=18)
ax[1].set_xlabel('Time (seconds)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/averager_waveforms.png')

########################################################################3
# Xk and Yk, then xt and yt, binary differencer
fig, ax = plt.subplots(2,1,figsize=(14,8))
X = 1-0.5*np.abs(np.arange(-5,6))/6
plot_spectrum(ax[0], X, 'X', True)

omega = np.arange(-int((len(X)-1)/2),int((len(X)+1)/2))*np.pi/int((len(X)+1)/2)
H = 2*np.abs(np.sin(omega/2))
plot_spectrum(ax[1], X*H, 'Y', True)
ax[0].set_title('Spectrum of $x[n]$',fontsize=18)
ax[1].set_title('Spectrum of $y[n]=h[n]\\ast x[n]$',fontsize=18)
ax[1].set_xlabel('Frequency (radians/sample)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/differencer_spectra.png')

fig, ax = plt.subplots(2,1,figsize=(14,8))
n_axis = np.linspace(0,36,300)
x = np.zeros(len(n_axis))
y = np.zeros(len(n_axis))
for k,w in enumerate(omega):
    x += np.real(X[k]*np.exp(1j*w*n_axis))
    y += np.real(H[k]*X[k]*np.exp(1j*w*n_axis))
ax[0].plot(n_axis,x)
ax[1].plot(n_axis,y)
ax[0].set_title('Input Waveform $x(t)$',fontsize=18)
ax[1].set_title('Output Waveform $y(t)$',fontsize=18)
ax[1].set_xlabel('Time (seconds)',fontsize=18)
fig.tight_layout()
fig.savefig('exp/differencer_waveforms.png')

######################################################################################
# Convolution figures

fig, ax = plt.subplots(2,1,figsize=(14,8))
n_lo = np.arange(40)
n_hi = np.linspace(0,40,400)
x_lo = np.cos(2*np.pi*n_lo/20)
x_hi = np.cos(2*np.pi*n_hi/20)
h = np.ones(5)
plot_convolution(ax, n_lo, n_hi, x_lo, x_hi, h, 5, 'exp/averager_lowfreq')

fig, ax = plt.subplots(2,1,figsize=(14,8))
h = np.array([1,-1])
plot_convolution(ax, n_lo, n_hi, x_lo, x_hi, h, 1, 'exp/differencer_lowfreq')

fig, ax = plt.subplots(2,1,figsize=(14,8))
x_lo = np.cos(2*np.pi*n_lo/7)
x_hi = np.cos(2*np.pi*n_hi/7)
h = np.ones(5)
plot_convolution(ax, n_lo, n_hi, x_lo, x_hi, h, 5, 'exp/averager_highfreq')

fig, ax = plt.subplots(2,1,figsize=(14,8))
h = np.array([1,-1])
plot_convolution(ax, n_lo, n_hi, x_lo, x_hi, h, 1, 'exp/differencer_highfreq')

