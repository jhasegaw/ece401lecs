import numpy as np
import matplotlib.figure, subprocess, os, wave

os.makedirs('exp',exist_ok=True)
################################################################################
# Probably globally useful stuff
N = 32
N2 = 2*N
N3 = 3*N
N4 = 4*N
nset = np.arange(-N2,N2)
delta = np.zeros(N4)
delta[nset==0] = 1
xwav = np.zeros(N4)
xwav[N2:N3] = np.random.randn(N)
hwav = np.zeros(N4)
hwav[N2:N3] = np.power(0.95,np.arange(N))*np.sin(np.pi*np.arange(N)/5)
ywav = np.convolve(xwav,hwav,'same')
xfft = np.fft.fft(xwav)
hfft = np.fft.fft(hwav)
yfft = np.fft.fft(ywav)
xcor = np.convolve(xwav,np.flip(xwav),'same')/N
hcor = np.convolve(hwav,np.flip(hwav),'same')
ycor = np.convolve(ywav,np.flip(ywav),'same')/N
xpow = np.square(np.abs(xfft))/N
hpow = np.square(np.abs(hfft))
ypow = np.square(np.abs(yfft))/N

omega = np.linspace(0,np.pi,N2)
xticks = np.pi*np.arange(5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
def plot_specs(axs, xspecs):
    x = [ xspec[:N2] for xspec in xspecs ]
    for k in range(3):
        axs[k].clear()
        axs[k].plot(omega,np.zeros(len(omega)),'k-',[0,1e-6],[min(0,np.amin(x[k]))-0.1,np.amax(x[k])+0.1],'k-')
        axs[k].plot(omega,x[k])
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels)
    axs[2].set_xlabel('$\omega$ (radians/sample)')

def plot_convolution(axs, x, h, y, n):
    '''
    h should be defined with the same resolution as nset, but not the same range:
    its range should be from 0 to max(nset)-min(nset).
    y should start out all zeros, but should be accumulated over time (as output of this func).
    Actually, if nset not integers, stem will give weird response.
    '''
    axs[0].clear()
    axs[0].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(x)-0.1,np.amax(x)+0.1],'k-')
    axs[0].stem(nset,x,use_line_collection=True)
    axs[1].clear()
    axs[1].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(h)-0.1,np.amax(h)+0.1],'k-')
    axs[1].stem(nset[:min(N4,n)],h[min(N4-1,n-1)::-1],use_line_collection=True)
    axs[2].clear()
    axs[2].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].stem(nset[:min(N4,n)],y[:min(N4,n)],use_line_collection=True)
    axs[2].set_xlabel('Time (Samples)')

def plot_correlation(axs, x, y, n):
    '''
    h should be defined with the same resolution as nset, but not the same range:
    its range should be from 0 to max(nset)-min(nset).
    y should start out all zeros, but should be accumulated over time (as output of this func).
    Actually, if nset not integers, stem will give weird response.
    '''
    axs[0].clear()
    axs[0].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(x)-0.1,np.amax(x)+0.1],'k-')
    axs[0].stem(nset,x,use_line_collection=True)
    axs[1].clear()
    axs[1].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(x)-0.1,np.amax(x)+0.1],'k-')
    if n <= N2:
        axs[1].stem(nset[:(N2+n)],x[(N2-n):],use_line_collection=True)
    else:
        axs[1].stem(nset,np.concatenate((np.zeros(n-N2),x[:(N4-(n-N2))])),use_line_collection=True)
    axs[2].clear()
    axs[2].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].stem(nset[:n],y[:n],use_line_collection=True)
    axs[2].set_xlabel('Time (Samples)')

fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)

###########################################################################
# Autocorrelation, power spectrum, and expected power spectrum of x
plot_convolution(axs,xwav,np.flip(xwav),delta,N4)
axs[0].set_title('$x[n]$')
axs[1].set_title('$x[n]$')
axs[2].set_title('$E[r_{xx}[n]]=\delta[n]$')
fig.tight_layout()
for n in range(1,N2):
    plot_correlation(axs,xwav,xcor,2*n)
    axs[0].set_title('$x[n]$')
    if 2*n < N2:
        axs[1].set_title('$x[n+%d]$'%(N2-2*n))
    else:
        axs[1].set_title('$x[n-%d]$'%(2*n-N2))
    axs[2].set_title('$r_{xx}[%d]=$correlate($x[n],x[n]$)')
    fig.tight_layout()
    fig.savefig('exp/xcor%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/xcor?.png exp/xcor??.png exp/xcor.gif'.split())
    
plot_specs(axs, [np.real(xfft), np.imag(xfft), xpow])
axs[0].set_title('Real($X(\omega)$)')
axs[1].set_title('Imag($X(\omega)$)')
axs[2].set_title('$R_{xx}(\omega)=|X(\omega)|^2/N$')
fig.tight_layout()
fig.savefig('exp/xpow.png')

plot_specs(axs, [np.zeros(N4), np.zeros(N4), np.ones(N4)])
axs[0].set_title('E[Real($X(\omega)$)]=0')
axs[1].set_title('E[Imag($X(\omega)$)]=0')
axs[2].set_title('E[$R_{xx}(\omega)=|X(\omega)|^2/N$]=1')
fig.tight_layout()
fig.savefig('exp/xexp.png')

###########################################################################
# Convolution of x and h to create y
plot_convolution(axs,xwav,np.flip(hwav),np.zeros(N4),N4)
axs[0].set_title('$x[n]$')
axs[1].set_title('$h[n]$')
axs[2].set_title('$E[x[n]*h[n]]=0$')
fig.tight_layout()
for n in range(0,int(N/2)):
    fig.savefig('exp/xhy%d.png'%(n))
for n in range(int(N/2),N2):
    plot_convolution(axs,xwav,np.concatenate((hwav[N2:],hwav[:N2])),ywav,2*n)
    axs[0].set_title('$x[n]$')
    axs[1].set_title('$h[%d-n]$'%(2*n-N2))
    axs[2].set_title('$y[%d]=$convolve($x[n],h[n]$)')
    fig.tight_layout()
    fig.savefig('exp/xhy%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/xhy?.png exp/xhy??.png exp/xhy.gif'.split())
    
plot_specs(axs, [xpow, hpow, ypow])
axs[0].set_title('$R_{xx}(\omega)$')
axs[1].set_title('$|H(\omega)|^2$')
axs[2].set_title('$R_{yy}(\omega)$')
fig.tight_layout()
fig.savefig('exp/allspecs.png')

###########################################################################
# Autocorrelation, power spectrum, and expected power spectrum of h
plot_convolution(axs,ywav,np.flip(ywav),hcor,N4)
axs[0].set_title('$h[n]$')
axs[1].set_title('$h[n]$')
axs[2].set_title('$r_{hh}[n]=$correlate($h[n],h[n]$)')
fig.tight_layout()
for n in range(1,N2):
    plot_correlation(axs,hwav,hcor,2*n)
    axs[0].set_title('$h[n]$')
    if 2*n < N2:
        axs[1].set_title('$h[n+%d]$'%(N2-2*n))
    else:
        axs[1].set_title('$h[n-%d]$'%(2*n-N2))
    axs[2].set_title('$r_{hh}[%d]=$correlate($h[n],h[n]$)')
    fig.tight_layout()
    fig.savefig('exp/hcor%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/hcor?.png exp/hcor??.png exp/hcor.gif'.split())
    
plot_specs(axs, [np.real(yfft), np.imag(yfft), ypow])
axs[0].set_title('Real($H(\omega)$)')
axs[1].set_title('Imag($H(\omega)$)')
axs[2].set_title('$|H(\omega)|^2$')
fig.tight_layout()
fig.savefig('exp/hpow.png')

###########################################################################
# Autocorrelation, power spectrum, and expected power spectrum of y
plot_convolution(axs,ywav,np.flip(ywav),hcor,N4)
axs[0].set_title('$y[n]$')
axs[1].set_title('$y[n]$')
axs[2].set_title('$E[r_{yy}[n]]=$correlate($h[n],h[n]$)')
fig.tight_layout()
for n in range(1,N2):
    plot_correlation(axs,ywav,ycor,2*n)
    axs[0].set_title('$y[n]$')
    if 2*n < N2:
        axs[1].set_title('$y[n+%d]$'%(N2-2*n))
    else:
        axs[1].set_title('$y[n-%d]$'%(2*n-N2))
    axs[2].set_title('$r_{yy}[%d]=$correlate($y[n],y[n]$)')
    fig.tight_layout()
    fig.savefig('exp/ycor%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/ycor?.png exp/ycor??.png exp/ycor.gif'.split())
    
plot_specs(axs, [np.real(yfft), np.imag(yfft), ypow])
axs[0].set_title('Real($Y(\omega)$)')
axs[1].set_title('Imag($Y(\omega)$)')
axs[2].set_title('$R_{yy}(\omega)=|Y(\omega)|^2/N$')
fig.tight_layout()
fig.savefig('exp/ypow.png')

plot_specs(axs, [np.zeros(N4), np.zeros(N4), hpow])
axs[0].set_title('E[Real($Y(\omega)$)]=0')
axs[1].set_title('E[Imag($Y(\omega)$)]=0')
axs[2].set_title('E$[R_{yy}(\omega)=|Y(\omega)|^2/N]=|H(\omega)|^2$')
fig.tight_layout()
fig.savefig('exp/yexp.png')


