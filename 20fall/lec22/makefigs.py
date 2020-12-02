import numpy as np
import matplotlib.figure, subprocess, os, wave

os.makedirs('exp',exist_ok=True)
################################################################################
# Probably globally useful stuff
def plot_axlines(ax, xlim, ylim):
    ax.plot([np.amin(xlim),np.amax(xlim)],[0,0],'k-',[0,1e-6],1.3*np.array([np.amin(ylim),np.amax(ylim)]),'k-')
    
###########################################################################
#  Image with three frames: impulse train x, ringing h, convolution y overstepping the DFT limit
#  length 100, period 25, h length 20, in a 200-sample axis
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,1)
x = np.zeros(100)
x[12:100:25] = 1
axs[0].stem(np.arange(100),x,use_line_collection=True)
plot_axlines(axs[0],[-50,150],[-0.2,1])
axs[0].set_title('$x[n]$ as  a finite-length signal')
h = np.exp(-0.05*np.arange(20))*np.cos(np.pi*np.arange(20)/5)
axs[1].stem(np.arange(20),h,use_line_collection=True)
plot_axlines(axs[1],[-50,150],h)
axs[1].set_title('Impulse  Response $h[n]$')
y = np.convolve(x,h,'full')
axs[2].stem(np.arange(119),y,use_line_collection=True)
axs[2].set_title('Output of linear  convolution, $y[n]=h[n]*x[n]$')
plot_axlines(axs[2],[-50,150],[-1,1])
fig.tight_layout()
fig.savefig('exp/pulsetrain_linear.png')

###########################################################################
#  Image with three frames: impulse train x, ringing h, circular convolution y
#  length 100, period 25, h length 20, in a 200-sample axis
y = np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(h,100)))
axs[2].clear()
axs[2].stem(np.arange(100),y,use_line_collection=True)
axs[2].set_title('Output of circular convolution, $y[n]$')
plot_axlines(axs[2],[-50,150],[-1,1])
fig.tight_layout()
fig.savefig('exp/pulsetrain_circular.png')

###########################################################################
#  Image with three frames: periodic impulse train x, ringing h, periodic circular convolution y
#  length 100, period 25, h length 20, in a 300-sample axis
nset = np.arange(-50,150)
x = np.zeros(nset.shape)
x[nset % 25 == 12] = 1
axs[0].stem(nset,x,use_line_collection=True)
plot_axlines(axs[0],[-50,150],[-0.2,1])
axs[0].set_title('$x[n]$ as an infinite-length periodic signal')
h = np.exp(-0.1*np.arange(20))*np.cos(np.pi*np.arange(20)/5)
axs[1].stem(np.arange(20),h,use_line_collection=True)
plot_axlines(axs[1],[-50,150],[-1,1])
axs[1].set_title('Impulse Response $h[n]$')
y = np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(h,200)))
axs[2].stem(nset,y,use_line_collection=True)
axs[2].set_title('Filter Output $y[n]$')
plot_axlines(axs[2],[-50,150],[-1,1])
fig.tight_layout()
fig.savefig('exp/pulsetrain_periodic.png')
  
###########################################################################
# Show x[n] as a delayed impulse, |X[k]|, |Y[k]|, and y[n] with
# circular convolution results.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,2)
nset = np.arange(32)
x = np.zeros(nset.shape)
x[25] = 1
x[30]=1
X = np.fft.fft(x)
Y = np.zeros(len(X),dtype=complex)
Y[:6] = X[:6]
Y[-5:] = X[-5:]
#h = np.fft.fftshift(np.sin(np.pi*(nset-15.5)/5)/(np.pi*(nset-15.5)))
#H = np.fft.fft(h)
#Y = H*X
y = np.real(np.fft.ifft(Y))
axs[0,0].stem(np.abs(X),use_line_collection=True)
plot_axlines(axs[0,0],[0,32],np.abs(X))
axs[0,0].set_title('Original Signal  $|X[k]|$, very broadband')
axs[0,1].stem(x,use_line_collection=True)
plot_axlines(axs[0,1],[0,32],x)
axs[0,1].set_title('Original Signal $x[n]$')
axs[1,0].stem(np.abs(Y),use_line_collection=True)
plot_axlines(axs[1,0],[0,32],np.abs(Y))
axs[1,0].set_title('Lowpass Filtered Signal $|Y[k]|$')
axs[1,0].set_xlabel('Frequency sample $k$')
axs[1,1].stem(y,use_line_collection=True)
plot_axlines(axs[1,1],[0,32],y)
axs[1,1].set_title('Lowpass filtering $y[n]$ using the DFT: time-domain aliasing')
axs[1,1].set_xlabel('Time domain sample $n$')
fig.tight_layout()
fig.savefig('exp/convolution_circular.png')

###########################################################################
# Show picture of y[n]=h[n]*x[n] with length M+L-1
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,1)
N=32
axs[0].stem(nset,x,use_line_collection=True)
plot_axlines(axs[0],[-16,48],x)
axs[0].set_title('$x[n]$ has length $L=32$')
h = np.real(np.fft.fftshift(np.fft.ifft(np.concatenate((np.ones(6),np.zeros(21),np.ones(5))))))
axs[1].stem(nset-16,h,use_line_collection=True)
plot_axlines(axs[1],[-16,48],h)
axs[1].set_title('$h[n]$ has length $M=32$')
y = np.convolve(h,x,mode='full')
axs[2].stem(np.arange(len(y))-16,y,use_line_collection=True)
axs[2].set_title('$y[n]=h[n]*x[n]$ has length $L+M-1=63$')
plot_axlines(axs[2],[-16,48],y)
fig.tight_layout()
fig.savefig('exp/convolution_linear.png')

###########################################################################
# Show picture of x[n], h[n], y[n], X[k], H[k], Y[k] zero-padded out
# to sufficient length so that circular convolution = linear.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,2)
nset = np.arange(32)
x = np.zeros(63)
x[25] = 1
x[30]=1
X = np.fft.fft(x)
Y = np.zeros(len(X),dtype=complex)
Y[:12] = X[:12]
Y[-11:] = X[-11:]
#h = np.zeros(63)
#h[:32] = np.fft.fftshift(np.sin(np.pi*(nset-15.5)/5)/(np.pi*(nset-15.5)))
#H = np.fft.fft(h)
#Y = H*X
y = np.real(np.fft.ifft(Y))
axs[0,0].stem(np.abs(X),use_line_collection=True)
plot_axlines(axs[0,0],[0,32],np.abs(X))
axs[0,0].set_title('$|X[k]|$ computed with $N=63$')
axs[0,1].stem(x,use_line_collection=True)
plot_axlines(axs[0,1],[0,32],x)
axs[0,1].set_title('$x[n]$ zero-padded to length $N=L+M-1=63$')
axs[1,0].stem(np.abs(Y),use_line_collection=True)
plot_axlines(axs[1,0],[0,32],np.abs(Y))
axs[1,0].set_title('$|Y[k]|$ computed with $N=63$')
axs[1,0].set_xlabel('Frequency sample $k$')
axs[1,1].stem(y,use_line_collection=True)
plot_axlines(axs[1,1],[0,32],y)
axs[1,1].set_title('$y[n]$: circular convolution using zero-padding to length $L+M-1$')
axs[1,1].set_xlabel('Time domain sample $n$')
fig.tight_layout()
fig.savefig('exp/convolution_zeropadded.png')

###########################################################################
# Picture showing three different $w_R[n]$, and corresponding $W_R(\omega)$
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,2)
N = 32
nset = np.arange(N)
L = [ 4, 8, 16, 32 ]
wR = [ np.concatenate((np.ones(L[p]),np.zeros(N-L[p]))) for p in range(4) ]
WRomega = [ np.abs(np.fft.fft(wR[k],8*N)) for k in range(4) ]
for row in range(3):
    axs[row,0].stem(nset,wR[row],use_line_collection=True)
    plot_axlines(axs[row,0],nset,wR[row])
    axs[row,0].set_title('Rectangular Window with length $L=%d$'%(L[row]))
    axs[row,1].plot(np.arange(8*N),WRomega[row])
    plot_axlines(axs[row,1],[0,8*N],[0,np.amax(WRomega[row])])
    axs[row,1].set_xticks(np.arange(9)*N)
    axs[row,1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
    axs[row,1].set_title('$|W_R(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/rectangular_dtft.png')

###########################################################################
# Picture showing three different $w[n]$, and corresponding $W_R[k]$ superimposed on $W_R(\omega)$
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,2)
N = 32
nset = np.arange(N)
L = [ 4, 8, 16, 32 ]
wR = [ np.concatenate((np.ones(L[p]),np.zeros(N-L[p]))) for p in range(4) ]
WRomega = [ np.abs(np.fft.fft(wR[k],8*N)) for k in range(4) ]
for row in range(3):
    axs[row,0].stem(nset,wR[row],use_line_collection=True)
    plot_axlines(axs[row,0],nset,wR[row])
    axs[row,0].set_title('Rectangular Window with length $L=%d$'%(L[row]))
    axs[row,1].plot(np.arange(8*N),WRomega[row])
    axs[row,1].stem(8*np.arange(N),WRomega[row][::8],use_line_collection=True)
    plot_axlines(axs[row,1],[0,8*N],[0,np.amax(WRomega[row])])
    axs[row,1].set_xticks(np.arange(9)*N)
    axs[row,1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
    axs[row,1].set_title('$|W_R(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/rectangular_dft.png')

###########################################################################
# Picture showing $w_R[n]$, and corresponding $W_R[k]$ superimposed on $W_R(\omega)$, when $L=N$
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
row = 3
axs[0].stem(nset,wR[row],use_line_collection=True)
plot_axlines(axs[0],nset,wR[row])
axs[0].set_title('Rectangular Window with length $L=%d$'%(L[row]))
axs[1].plot(np.arange(8*N),WRomega[row])
axs[1].stem(8*np.arange(N),WRomega[row][::8],use_line_collection=True)
plot_axlines(axs[1],[0,8*N],[0,np.amax(WRomega[row])])
axs[1].set_xticks(np.arange(9)*N)
axs[1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
axs[1].set_title('$|W_R(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/rectangular_fulllength.png')

###########################################################################
# Picture showing $x[n]$, and corresponding $X[k]$ superimposed on $X(\omega)$, for three
# different lengths of the rectangular window.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(4,2)
N = 32
nset = np.arange(N)
L = [ 4, 8, 16, 32 ]
for row in range(4):
    nset = np.arange(L[row])-0.5*(L[row]-1)
    omegac = np.pi/2
    wR[row][:L[row]] = np.sin(omegac*nset)/(np.pi*nset)
WRomega = [ np.abs(np.fft.fft(wR[k],8*N)) for k in range(4) ]
for row in range(4):
    axs[row,0].stem(np.arange(N),wR[row],use_line_collection=True)
    plot_axlines(axs[row,0],nset,wR[row])
    axs[row,0].set_title('Windowed LPF $x[n]$, $L=%d$'%(L[row]))
    axs[row,1].plot(np.arange(8*N),WRomega[row])
    axs[row,1].stem(8*np.arange(N),WRomega[row][::8],use_line_collection=True)
    plot_axlines(axs[row,1],[0,8*N],[0,np.amax(WRomega[row])])
    axs[row,1].set_xticks(np.arange(9)*N)
    axs[row,1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
    axs[row,1].set_title('$|X(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/rectangular_windowed.png')

###########################################################################
# Picture showing $w_H[n]$, and corresponding $W_H[k]$ superimposed on
# $W_H(\omega)$, for three different $L$.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,2)
N = 32
nset = np.arange(N)
L = [ 4, 8, 16, 32 ]
wR = [ np.concatenate((np.hamming(L[p]),np.zeros(N-L[p]))) for p in range(4) ]
WRomega = [ np.abs(np.fft.fft(wR[k],8*N)) for k in range(4) ]
for row in range(3):
    axs[row,0].stem(nset,wR[row],use_line_collection=True)
    plot_axlines(axs[row,0],nset,wR[row])
    axs[row,0].set_title('Hamming Window with length $L=%d$'%(L[row]))
    axs[row,1].plot(np.arange(8*N),WRomega[row])
    axs[row,1].stem(8*np.arange(N),WRomega[row][::8],use_line_collection=True)
    plot_axlines(axs[row,1],[0,8*N],[0,np.amax(WRomega[row])])
    axs[row,1].set_xticks(np.arange(9)*N)
    axs[row,1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
    axs[row,1].set_title('$|W_H(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/hamming_dft.png')

###########################################################################
# Picture showing $x[n]$, and corresponding $X[k]$ superimposed on $X(\omega)$, after windowing
# by three different lengths of Hamming window.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(4,2)
N = 32
L = [ 4, 8, 16, 32 ]
for row in range(4):
    nset = np.arange(L[row])-0.5*(L[row]-1)
    omegac = np.pi/2
    wR[row][:L[row]] = np.hamming(L[row])*np.sin(omegac*nset)/(np.pi*nset)
WRomega = [ np.abs(np.fft.fft(wR[k],8*N)) for k in range(4) ]
for row in range(4):
    axs[row,0].stem(np.arange(N),wR[row],use_line_collection=True)
    plot_axlines(axs[row,0],np.arange(N),wR[row])
    axs[row,0].set_title('Hamming-Windowed LPF $x[n]$, $L=%d$'%(L[row]))
    axs[row,1].plot(np.arange(8*N),WRomega[row])
    axs[row,1].stem(8*np.arange(N),WRomega[row][::8],use_line_collection=True)
    plot_axlines(axs[row,1],[0,8*N],[0,np.amax(WRomega[row])])
    axs[row,1].set_xticks(np.arange(9)*N)
    axs[row,1].set_xticklabels(['0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/4','2π'])
    axs[row,1].set_title('$|X(\omega)|$, $L=%d$'%(L[row]))
fig.tight_layout()
fig.savefig('exp/hamming_windowed.png')

###########################################################################
# Picture showing x[n]=cos(8n/N) finite-length, periodic repetition of x[n],
# and |X[k]|
fig=matplotlib.figure.Figure((14,6))
axs=fig.subplots(3,1)
n1 = np.arange(32)
omega0 = 2*np.pi/8
x1 = np.cos(omega0*n1)
n2 = np.arange(-16,48)
x2 = np.cos(omega0*n2)
axs[0].stem(n1,x1,use_line_collection=True)
plot_axlines(axs[0],n2,x2)
axs[0].set_title('Finite length $x[n]$: $N=4T_0$')
axs[0].set_xlabel('Time domain samples, $n$')
axs[1].stem(n2,x2,use_line_collection=True)
plot_axlines(axs[1],n2,x2)
axs[1].set_title('Periodic $x[n]$')
axs[1].set_xlabel('Time domain samples, $n$')
X1 = np.abs(np.fft.fft(x1))
axs[2].stem(n1,X1,use_line_collection=True)
plot_axlines(axs[2],n1,[0,np.amax(X1)])
axs[2].set_title('$|X[k]|$: impulses at $k=4$ and $k=N-4$')
axs[2].set_xlabel('Frequency domain samples, $k$')
fig.tight_layout()
fig.savefig('exp/puretone_integer.png')

###########################################################################
# Picture showing x[n]=cos(7.5n/N) finite-length, periodic repetition of x[n],
# and |X[k]|
fig=matplotlib.figure.Figure((14,6))
axs=fig.subplots(3,1)
n1 = np.arange(32)
omega0 = 2*3.5*np.pi/32
x1 = np.cos(omega0*n1)
n2 = np.arange(-16,48)
x2 = np.concatenate((x1[16:],x1,x1[:16]))
axs[0].stem(n1,x1,use_line_collection=True)
plot_axlines(axs[0],n2,x2)
axs[0].set_title('Finite length $x[n]$: $N=3.5T_0$')
axs[0].set_xlabel('Time domain samples, $n$')
axs[1].stem(n2,x2,use_line_collection=True)
axs[1].set_xlabel('Time domain samples, $n$')
plot_axlines(axs[1],n2,x2)
axs[1].set_title('Periodic $x[n]$')
X1 = np.abs(np.fft.fft(x1))
axs[2].stem(n1,X1,use_line_collection=True)
plot_axlines(axs[2],n1,[0,np.amax(X1)])
axs[2].set_title('$|X[k]|=|W_R(\omega)|$ shifted to $k=3.5$ and $k=N-3.5$')
axs[2].set_xlabel('Frequency domain samples, $k$')
fig.tight_layout()
fig.savefig('exp/puretone_noninteger.png')

###########################################################################
# Show a picture of X(omega) overlaid on X[k] for a pure tone whose period is a sub-multiple
# of N
fig=matplotlib.figure.Figure((14,6))
axs=fig.subplots(3,1)
n1 = np.arange(32)
omega0 = 2*np.pi/8
x1 = np.cos(omega0*n1)
n2 = np.arange(-16,48)
x2 = np.cos(omega0*n2)
axs[0].stem(n1,x1,use_line_collection=True)
plot_axlines(axs[0],n2,x2)
axs[0].set_title('Finite length $x[n]$: $N=4T_0$')
axs[0].set_xlabel('Time domain samples, $n$')
axs[1].stem(n2,x2,use_line_collection=True)
plot_axlines(axs[1],n2,x2)
axs[1].set_title('Periodic $x[n]$')
axs[1].set_xlabel('Time domain samples, $n$')
X1 = np.real(np.fft.fft(x1))
X2 = np.real(np.fft.fft(np.concatenate((x1[:16],np.zeros(100-32),x1[16:]))))
axs[2].plot(np.arange(100)*32/100,X2)
axs[2].stem(n1,X1,use_line_collection=True)
plot_axlines(axs[2],n1,[np.amin(X1),np.amax(X1)])
axs[2].set_title('$X[k]=X(\omega_k)$: impulses at $k=4$ and $k=N-4$')
axs[2].set_xlabel('Frequency domain samples, $k$')
fig.tight_layout()
fig.savefig('exp/puretone_integerdtft.png')

###########################################################################
# Show a picture of X(omega) overlaid on X[k] for a pure tone whose period is NOT a sub-multiple
# of N
fig=matplotlib.figure.Figure((14,6))
axs=fig.subplots(3,1)
n1 = np.arange(32)
omega0 = 2*3.5*np.pi/32
x1 = np.cos(omega0*n1)
n2 = np.arange(-16,48)
x2 = np.concatenate((x1[16:],x1,x1[:16]))
axs[0].stem(n1,x1,use_line_collection=True)
axs[0].set_xlabel('Time domain samples, $n$')
plot_axlines(axs[0],n2,x2)
axs[0].set_title('Finite length $x[n]$: $N=3.5T_0$')
axs[1].stem(n2,x2,use_line_collection=True)
axs[1].set_xlabel('Time domain samples, $n$')
plot_axlines(axs[1],n2,x2)
axs[1].set_title('$x[n]$ periodic with period $N=32$, NOT with period $T_0$!')
X1 = np.abs(np.fft.fft(x1))
X2 = np.abs(np.fft.fft(np.concatenate((x1[:16],np.zeros(100-31),x1[17:]))))
axs[2].plot(np.arange(100)*32/100,X2)
axs[2].stem(n1,X1,use_line_collection=True)
plot_axlines(axs[2],n1,[np.amin(X1),np.amax(X1)])
axs[2].set_title('$|X[k]|=|W_R(\omega)|$ shifted to $k=3.5$ and $k=N-3.5$')
axs[2].set_xlabel('Frequency domain samples, $k$')
fig.tight_layout()
fig.savefig('exp/puretone_nonintegerdtft.png')
