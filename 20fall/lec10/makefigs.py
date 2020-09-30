import numpy  as np
import matplotlib.figure, subprocess

def plotspec(ax,omega,X,xticks,xticklabels):
    ax.plot(omega,np.zeros(len(omega)),'k-') # omega axis
    ax.plot([0,1e-6],[np.amin(X)-0.1,np.amax(X)+0.1],'k-') # X axis
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(np.amin(omega),np.amax(omega))
    ax.set_ylim(np.amin(X)-0.1,np.amax(X)+0.1)
    ax.plot(omega,X,'b-')

N = 64

#############################################################################################
# ideal LPF X,H,Y
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1)
brownian = np.convolve(np.random.randn(N),[1,0.5],mode='same')
X = np.fft.fftshift(np.fft.fft(brownian))
omega = np.linspace(-np.pi,np.pi,N,endpoint=False)
LI = np.zeros(N)
LI[np.abs(omega)<=np.pi/4] = 1
xticks = np.pi*np.array([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
xticklabels = ['-π','-3π/4','-π/2','-π/4','0','π/4','π/2','3π/4','π']
plotspec(axs[0],omega,np.abs(X), xticks, xticklabels)
axs[0].set_title('$|X(\omega)|$')
plotspec(axs[1],omega,LI, xticks, xticklabels)
axs[1].set_title('$L_I(\omega)$')
plotspec(axs[2],omega,np.abs(X)*LI, xticks, xticklabels)
axs[2].set_title('$|Y(\omega)|=L_I(\omega)|X(\omega)|$')
fig.tight_layout()
fig.savefig('exp/ideal_lpf.png')

#############################################################################################
# ideal LPF three cutoffs
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,2)
nset = np.arange(-(N-1),N)
l = np.zeros(len(nset))
for c in range(3):
    cutoff = (c+1)*np.pi/4
    L = np.zeros(N)
    L[np.abs(omega)<=cutoff]=1
    plotspec(axs[c,0],omega,L, xticks, xticklabels)
    axs[c,0].set_title('$L_I(\omega)$, cutoff=%s'%(xticklabels[5+c]))
    l[nset==0]=cutoff/np.pi
    l[nset != 0] = np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
    axs[c,1].plot(nset,l,nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.1,1.1],'k-')
    axs[c,1].set_title('$l_I[n]$, cutoff=%s'%(xticklabels[5+c]))
    axs[c,1].set_xlim(-10,10)
    axs[c,1].set_ylim(np.amin(l)-0.1,np.amax(l)+0.1)
fig.tight_layout()
fig.savefig('exp/ideal_lpf_threecutoffs.png')


#############################################################################################
# Video showing convolution of a noisy input signal with an ideal LPF.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
cutoff = np.pi/2
l[nset==0]=cutoff/np.pi
l[nset != 0] = np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
y = np.convolve(brownian, l, mode='valid')
axs[0].plot(brownian)
axs[0].plot([0,N-1],[0,0],'k-',[0,1e-6],[np.amin(brownian)-0.1,np.amax(brownian)+0.1],'k-')
axs[0].set_title('Noisy $x[m]$')
for n in range(N):
    axs[1].clear()
    axs[1].plot(l[(N-1+n):(n-1):-1])
    axs[1].plot([0,N-1],[0,0],'k-',[0,1e-6],[-0.5,0.5],'k-')
    axs[1].set_title('$l_I[%d-m]$'%(n))
    axs[2].clear()
    axs[2].plot(np.arange(0,n+1),y[:(n+1)],'b-')
    axs[2].plot([0,N-1],[0,0],'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].set_title('$y[n]=l_I[n]*x[n]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    fig.savefig('exp/ideal_lpf_convolution%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/ideal_lpf_convolution?.png exp/ideal_lpf_convolution??.png exp/ideal_lpf_convolution.gif'.split())

#############################################################################################
# Image showing ideal HPF
fig = matplotlib.figure.Figure((5,4))
ax = fig.subplots()
H = np.zeros(N)
H[np.abs(omega)>=np.pi/2] = 1
plotspec(ax,omega,H,xticks,xticklabels)
ax.set_title('$H_I(\omega)$')
fig.savefig('exp/ideal_hpf.png')

#############################################################################################
# Video showing $H_I(\omega)$, starting at domain $(-\pi,\pi)$, zooming out to show $(-3\pi,3\pi)$.
superomega = np.linspace(-3*np.pi,3*np.pi,N)
superxticks = np.pi*np.arange(-6,7)/2
superxticklabels = ['-3π','-5π/2','-2π','-3π/2','-π','-π/2','0','π/2','π','3π/2','2π','5π/2','3π']
H = np.ones(len(superomega))
H[np.abs(superomega)<=np.pi/3]=0
H[np.abs(np.abs(superomega)-2*np.pi)<=np.pi/3]=0
fig = matplotlib.figure.Figure((7,4))
ax = fig.subplots()
plotspec(ax,superomega,H,superxticks,superxticklabels)
ax.set_title('Ideal Highpass Filter')
for n in range(100):
    xlim = np.pi*(1+2*n/99)
    ax.set_xlim(-xlim,xlim)
    fig.savefig('exp/ideal_hpf_zoom%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/ideal_hpf_zoom?.png exp/ideal_hpf_zoom??.png exp/ideal_hpf_zoom.gif'.split())

#############################################################################################
# Video: showing $\cos(\omega n)$ as $\omega ranges from 0 to $2\pi$.
freqs = np.array([0,1,1,2,4,8,16,32,64])
freqs = np.concatenate((freqs,np.cumsum(freqs[::-1])))
for k in range(len(freqs)):
    ax.clear()
    ax.plot(nset,np.zeros(len(nset)),'k-',[0,1e-6],[-1.1,1.1],'k-')
    freq = freqs[k]/64
    ax.plot(nset,np.cos(np.pi*freq*nset))
    ax.set_title('cos(%2.2fπn)=cos(%2.2fπn)'%(freq,2-freq))
    fig.savefig('exp/cosine_sweep%d.png'%(k))
subprocess.call('convert -delay 33 -dispose previous exp/cosine_sweep?.png exp/cosine_sweep??.png exp/cosine_sweep.gif'.split())

#############################################################################################
# ideal HPF three cutoffs
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,2)
nset = np.arange(-(N-1),N)
h = np.zeros(len(nset))
for c in range(3):
    cutoff = (c+1)*np.pi/4
    H = np.zeros(N)
    H[np.abs(omega)>=cutoff]=1
    plotspec(axs[c,0],omega,H,xticks,xticklabels)
    axs[c,0].set_title('H_I(\omega), cutoff=%s'%(xticklabels[5+c]))
    h[nset==0]=1 -cutoff/np.pi
    h[nset != 0] = -np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
    axs[c,1].plot(nset,h,nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.1,1.1],'k-')
    axs[c,1].set_title('$h_I[n]$, cutoff=%s'%(xticklabels[5+c]))
    axs[c,1].set_xlim(-10,10)
    axs[c,1].set_ylim(np.amin(h)-0.1,np.amax(h)+0.1)
fig.tight_layout()
fig.savefig('exp/ideal_hpf_threecutoffs.png')

#############################################################################################
# Video showing convolution of a noisy input signal with an ideal LPF.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
cutoff = np.pi/2
h[nset==0]= 1 - cutoff/np.pi
h[nset != 0] = -np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
y = np.convolve(brownian, h, mode='valid')
axs[0].plot(brownian)
axs[0].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[np.amin(brownian)-0.1,np.amax(brownian)+0.1],'k-')
axs[0].set_title('Noisy $x[m]$')
for n in range(N):
    axs[1].clear()
    axs[1].plot(h[((N-1)+n):(n-1):-1])
    axs[1].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[-0.5,0.5],'k-')
    axs[1].set_title('$h_I[%d-m]$'%(n))
    axs[2].clear()
    axs[2].plot(np.arange(0,n+1),y[:(n+1)],'b-')
    axs[2].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].set_title('$y[n]=h_I[n]*x[n]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    fig.savefig('exp/ideal_hpf_convolution%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/ideal_hpf_convolution?.png exp/ideal_hpf_convolution??.png exp/ideal_hpf_convolution.gif'.split())

#############################################################################################
# Image showing ideal BPF
fig = matplotlib.figure.Figure((5,4))
ax = fig.subplots()
B = np.zeros(N)
B[np.abs((np.abs(omega)-np.pi/3))<=np.pi/8] = 1
plotspec(ax,omega,B,xticks,xticklabels)
ax.set_title('$B_I(\omega)$')
fig.savefig('exp/ideal_bpf.png')

#############################################################################################
# ideal BPF three cutoffs
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,2)
nset = np.arange(-(N-1),N)
cutoffs = ['π/8','3π/8','5π/8','7π/8']
b = np.zeros(len(nset))
for c in range(2,-1,-1):   # Set it this way so b remains set for the video
    lo = (2*c+1)*np.pi/8    
    hi = (2*c+3)*np.pi/8    
    B = np.zeros(N)
    B[(np.abs(omega)>=lo)&(np.abs(omega)<=hi)]=1
    plotspec(axs[c,0],omega,B,xticks,xticklabels)
    axs[c,0].set_title('$B_I(\omega)$, cutoffs=%s, %s'%(cutoffs[c],cutoffs[c+1]))
    b[nset==0] = (hi-lo)/np.pi
    b[nset != 0] = (np.sin(hi*nset[nset!=0])-np.sin(lo*nset[nset!=0]))/(np.pi*nset[nset!=0])
    axs[c,1].plot(nset,b,nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.1,1.1],'k-')
    axs[c,1].set_title('$b_I[n]$, cutoffs=%s, %s'%(cutoffs[c],cutoffs[c+1]))
    axs[c,1].set_xlim(-10,10)
    axs[c,1].set_ylim(np.amin(b)-0.1,np.amax(b)+0.1)
fig.tight_layout()
fig.savefig('exp/ideal_bpf_threecutoffs.png')

#############################################################################################
# Video showing convolution of a noisy input signal with an ideal BPF.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
y = np.convolve(brownian, b, mode='valid')
axs[0].plot(brownian)
axs[0].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[np.amin(brownian)-0.1,np.amax(brownian)+0.1],'k-')
axs[0].set_title('Noisy $x[m]$')
for n in range(N):
    axs[1].clear()
    axs[1].plot(b[((N-1)+n):(n-1):-1])
    axs[1].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[-0.5,0.5],'k-')
    axs[1].set_title('$b_I[%d-m]$'%(n))
    axs[2].clear()
    axs[2].plot(np.arange(0,n+1),y[:(n+1)],'b-')
    axs[2].plot([0,(N-1)],[0,0],'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].set_title('$y[n]=b_I[n]*x[n]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    fig.savefig('exp/ideal_bpf_convolution%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/ideal_bpf_convolution?.png exp/ideal_bpf_convolution??.png exp/ideal_bpf_convolution.gif'.split())


#############################################################################################
# Image showing $l_I[n]$, $L_I(\omega)$, truncated $l[n]$, $L(\omega)$.
M = 9
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
cutoff = np.pi/4
li = np.zeros(len(nset))
li[nset==0] = cutoff/np.pi
li[nset!=0] = np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
LI = np.zeros(len(omega))
LI[np.abs(omega)<=cutoff]=1
plotspec(axs[0,1],omega,LI,xticks,xticklabels)
axs[0,1].set_title('$L_I(\omega)$, cutoff=%s'%(xticklabels[5]))
axs[0,0].plot(nset,li,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[0,0].set_title('$l_I[n]$, cutoff=%s'%(xticklabels[5]))
axs[0,0].set_xlim(-2*M,2*M)
axs[0,0].set_ylim(np.amin(li)-0.1,np.amax(li)+0.1)

narr = np.concatenate((np.arange(-2*M,-M+1),np.arange(-M,M+1),np.arange(M,2*M+1)))
l = np.concatenate((np.zeros(-M+1+2*M),li[(-M<=nset)&(nset<=M)],np.zeros(2*M+1-M)))
L = np.fft.fftshift(np.real(np.fft.fft(np.fft.fftshift(l))))
omeg = np.linspace(-np.pi,np.pi,len(L))
plotspec(axs[1,1],omeg,L,xticks,xticklabels)
axs[1,1].set_title('$L(\omega)$, cutoff=%s'%(xticklabels[5]))
axs[1,0].plot(narr,l,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[1,0].set_title('Truncated $l[n]$, cutoff=%s'%(xticklabels[5]))
axs[1,0].set_xlim(-2*M,2*M)
axs[1,0].set_ylim(np.amin(l)-0.1,np.amax(l)+0.1)
fig.savefig('exp/odd_truncated.png')

#############################################################################################
# Image showing truncated $l_I[n]$, Hamming window, and windowed $l[n]$.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)

axs[0].plot(narr,l,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[0].set_title('Truncated $l[n]$, cutoff=%s'%(xticklabels[5]))
axs[0].set_xlim(-2*M,2*M)
axs[0].set_ylim(np.amin(l)-0.1,np.amax(l)+0.1)
w = np.concatenate((np.zeros(-M+1+2*M),0.54+0.46*np.cos(np.pi*np.arange(-M,M+1)/M),np.zeros(2*M+1-M)))
axs[1].plot(narr,w,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.1,1.1],'k-')
axs[1].set_title('Hamming Window $w[n]$, Length=%d'%(2*M+1))
axs[2].plot(narr,w*l,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[2].set_title('Windowed Filter $w[n]l_I[n]$, Length=%d'%(2*M+1))
axs[2].set_xlim(-2*M,2*M)
axs[2].set_ylim(np.amin(l*w)-0.1,np.amax(l*w)+0.1)
fig.tight_layout()
fig.savefig('exp/odd_window.png')

#############################################################################################
# Image showing $l_I[n]$, $L_I(\omega)$, windowed $l[n]$, $L(\omega)$
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
cutoff = np.pi/4
li = np.zeros(len(nset))
li[nset==0] = cutoff/np.pi
li[nset!=0] = np.sin(cutoff*nset[nset != 0])/(np.pi*nset[nset != 0])
LI = np.zeros(len(omega))
LI[np.abs(omega)<=cutoff]=1
plotspec(axs[0,1],omega,LI,xticks,xticklabels)
axs[0,1].set_title('$L_I(\omega)$, cutoff=%s'%(xticklabels[5]))
axs[0,0].plot(nset,li,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[0,0].set_title('$l_I[n]$, cutoff=%s'%(xticklabels[5]))
axs[0,0].set_xlim(-2*M,2*M)
axs[0,0].set_ylim(np.amin(li)-0.1,np.amax(li)+0.1)
LW = np.fft.fftshift(np.real(np.fft.fft(np.fft.fftshift(l*w))))
plotspec(axs[1,1],omeg,LW,xticks,xticklabels)
axs[1,1].set_title('Windowed $L(\omega)$, cutoff=%s'%(xticklabels[5]))
axs[1,0].plot(narr,l*w,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[1,0].set_title('Windowed $l[n]$, cutoff=%s'%(xticklabels[5]))
axs[1,0].set_xlim(-2*M,2*M)
axs[1,0].set_ylim(np.amin(li)-0.1,np.amax(li)+0.1)
fig.tight_layout()
fig.savefig('exp/odd_windowed.png')

#############################################################################################
# Picture showing ideal LPF delayed by 9.5 samples, say.
M=10
fig = matplotlib.figure.Figure((5,4))
ax = fig.subplots()
ldel = np.sin(cutoff*(nset-M+0.5))/(np.pi*(nset-M+0.5))
ax.stem(nset,ldel,use_line_collection=True)
ax.plot(nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
ax.set_xlim(-0.5*M,2.5*M)
ax.set_title('Ideal LPF, delayed by %2.1f samples'%(M-0.5))
fig.savefig('exp/delayed_lpf.png')

#############################################################################################
# Picture showing ideal LPF delayed by 9.5 samples, magnitude spectrum, and phase spectrum.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1)
li = np.sin(cutoff*(nset-M+0.5))/(np.pi*(nset-M+0.5))
axs[0].stem(nset,li,use_line_collection=True)
axs[0].plot(nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[0].set_xlim(-0.5*M,2.5*M)
axs[0].set_ylim(np.amin(li)-0.1,np.amax(li)+0.1)
axs[0].set_title('Ideal LPF, delayed by %2.1f samples'%(M-0.5))
LI = np.zeros(len(omega))
LI[np.abs(omega)<=cutoff] = 1
plotspec(axs[1],omega,LI,xticks,xticklabels)
axs[1].set_title('Magnitude of delayed filter = $|L(\omega)|$')
phi = np.zeros(len(omega))
phi[np.abs(omega)<=cutoff]=-(M-0.5)*omega[np.abs(omega)<=cutoff]
plotspec(axs[2],omega,phi,xticks,xticklabels)
axs[2].set_title('Phase of delayed filter ∠$L(\omega)$= -j%2.1f$\omega$'%(M-0.5))
fig.tight_layout()
fig.savefig('exp/delayed_lpf_spectrum.png')

#############################################################################################
# Image showing truncated delayed $l_I[n]$, Hamming window, and windowed $l[n]$.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1)
narr = np.concatenate((np.arange(-10,1),np.arange(0,2*M),np.arange(2*M-1,30)))
l = np.concatenate((np.zeros(1+10),li[(0<=nset)&(nset<2*M)],np.zeros(30-(2*M-1))))
axs[0].plot(narr,l,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.25,0.25],'k-')
axs[0].set_xlim(-10,30)
axs[0].set_ylim(np.amin(l)-0.1,np.amax(l)+0.1)
axs[0].set_title('Truncated Delayed $l[n]$, cutoff=%s'%(xticklabels[5]))
w=np.concatenate((np.zeros(1+10),0.54-0.46*np.cos(2*np.pi*np.arange(0,2*M)/(2*M-1)),np.zeros(30-(2*M-1))))
axs[1].plot(narr,w,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.1,1],'k-')
axs[1].set_title('Hamming Window $w[n]$, Length=%d'%(2*M))
axs[1].set_xlim(-10,30)
axs[2].plot(narr,w*l,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.25,0.25],'k-')
axs[2].set_title('Windowed Delayed Filter $w[n]l_I[n-%2.1f]$, Length=%d'%(M-0.5,2*M+1))
axs[2].set_xlim(-10,30)
fig.tight_layout()
fig.savefig('exp/even_window.png')

#############################################################################################
# Picture showing delayed $l_I[n]$, $|L_I(\omega)|$, delayed windowed $l[n]$, $|L(\omega)|$.
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
cutoff = np.pi/4
plotspec(axs[0,1],omega,LI,xticks,xticklabels)
axs[0,1].set_title('$L_I(\omega)$, cutoff=%s'%(xticklabels[5]))
axs[0,0].plot(nset,li,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[0,0].set_title('$l_I[n]$, cutoff=%s'%(xticklabels[5]))
axs[0,0].set_xlim(-10,30)
LW = np.fft.fftshift(np.abs(np.fft.fft(np.fft.fftshift(l*w))))
omeg = np.linspace(-np.pi,np.pi,len(LW))
plotspec(axs[1,1],omeg,LW,xticks,xticklabels)
axs[1,1].set_title('Windowed $|L(\omega)|$, cutoff=%s'%(xticklabels[5]))
axs[1,0].plot(narr,l*w,'b-',nset,np.zeros(len(nset)),'k-',[0,1e-6],[-0.5,1],'k-')
axs[1,0].set_title('Windowed $l[n]$, cutoff=%s'%(xticklabels[5]))
axs[1,0].set_xlim(-10,30)
fig.tight_layout()
fig.savefig('exp/even_windowed.png')

#############################################################################################
# Video showing convolution of a noisy input signal with an even LPF.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
cutoff = np.pi/2
narr = np.arange(0,2*M)
l = li[(0<=nset)&(nset<2*M)]
w = 0.54-0.46*np.cos(2*np.pi*np.arange(0,2*M)/(2*M-1))
l = l*w
y = np.convolve(brownian, l)
axs[0].plot(brownian)
axs[0].plot([0,len(y)-1],[0,0],'k-',[0,1e-6],[np.amin(brownian)-0.1,np.amax(brownian)+0.1],'k-')
axs[0].set_title('Noisy $x[m]$')
axs[0].set_xlim(0,len(y)-1)
axs[0].set_ylim(np.amin(brownian)-0.1,np.amax(brownian)+0.1)
for n in range(len(y)):
    axs[1].clear()
    axs[1].plot(n-narr,l)
    axs[1].plot([0,len(y)-1],[0,0],'k-',[0,1e-6],[-0.5,0.5],'k-')
    axs[1].set_xlim(0,len(y)-1)
    axs[1].set_ylim(np.amin(l)-0.1,np.amax(l)+0.1)
    axs[1].set_title('$l[%d-m]$'%(n))
    axs[2].clear()
    axs[2].plot(np.arange(0,n),y[:n],'b-')
    axs[2].plot([0,len(y)-1],[0,0],'k-',[0,1e-6],[np.amin(y)-0.1,np.amax(y)+0.1],'k-')
    axs[2].set_title('$y[n]=l[n]*x[n]$')
    axs[2].set_xlim(0,len(y)-1)
    axs[2].set_ylim(np.amin(y)-0.1,np.amax(y)+0.1)
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    fig.savefig('exp/even_lpf_convolution%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/even_lpf_convolution?.png exp/even_lpf_convolution??.png exp/even_lpf_convolution.gif'.split())







