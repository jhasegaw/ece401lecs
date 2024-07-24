import numpy as np
import math,subprocess,os
import matplotlib.figure
import matplotlib.pyplot

os.makedirs('exp',exist_ok=True)

squarewave = np.array([1,1,1,0,0,0,0,0,0,1,1])
ntimes = 5
s = np.tile(squarewave,ntimes)
S = np.fft.fft(s)
nf = int(len(S)/2)+1
f = matplotlib.figure.Figure(figsize=(5, 4))
a = f.subplots(2,1)
a[0].stem(s)
a[0].set_title('Square Wave of length 11, $x[n]$')
a[0].set_xticks([0,11,22,33,44])
a[1].clear()
a[1].stem(np.real(S[:nf:ntimes])/ntimes)
a[1].set_title('Spectrum of the Square Wave, $X[k]$')
a[1].set_xticks([0,1,2,3,4,5])
a[1].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
a[1].set_xlabel('Frequency $\omega$ (radians/sample)')
f.tight_layout()
f.savefig('exp/squarewave_real.png')

#########################################################3
# pure delay
g = np.zeros(50)
g[5] = 1
G = np.fft.fft(g)

fig = matplotlib.figure.Figure(figsize=(5, 4))
axs = fig.subplots(3,1)
axs[0].stem(g)
axs[0].set_title('Impulse Response of a Pure-Delay Filter, $g[n]=\delta[n-5]$')
axs[1].plot(np.absolute(G[:26]))
axs[1].set_xticks([])
axs[1].set_title('Magnitude Response of a Pure-Delay Filter $|G(\omega)|=1$')
axs[2].plot(np.angle(G[:26]))
axs[2].set_xticks([0,5,10,15,20,25])
axs[2].set_xticklabels(['0','π/5','2π/5','3π/5','4π/5','π'])
axs[2].set_title('Phase Response of a  Pure-Delay Filter $∠G(\omega)=-5\omega$')
axs[2].set_ylim(-np.pi,np.pi)
axs[2].set_yticks([-np.pi,0,np.pi])
axs[2].set_yticklabels(['-π','0','π'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/puredelay.png')
                  

#########################################################3
# Square wave
squarewave = np.array([1,1,1,0,0,0,0,0,0,1,1])
ntimes = 5
s = np.tile(squarewave,ntimes)
S = np.fft.fft(s)
k = np.linspace(0,len(squarewave),len(S),endpoint=False)
nf = int(len(S)/2)+1
axs[0].clear()
axs[0].stem(s)
axs[0].set_title('Square Wave of length 11, $x[n]$')
axs[0].set_xticks([0,11,22,33,44])
axs[1].clear()
axs[1].stem(np.absolute(S[:nf:ntimes])/ntimes)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of the Square Wave, $|X[k]|$')
axs[2].clear()
axs[2].stem(np.angle(S[:nf:ntimes]))
axs[2].set_title('Phase Spectrum of the Square Wave, $∠X[k]$')
#axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticks([0,1,2,3,4,5])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_yticks([-np.pi,0,np.pi])
axs[2].set_yticklabels(['-π','0','π'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/squarewave.png')

#########################################################3
# Delayed Square wave
ds = np.roll(s,5)
DS = np.fft.fft(ds)
axs[0].clear()
axs[0].stem(ds)
axs[0].set_title('Delayed Square Wave of length 11, $y[n]=x[n-5]$')
axs[0].set_xticks([0,11,22,33,44])
axs[1].clear()
axs[1].stem(np.absolute(DS[:nf:ntimes])/ntimes)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of a Delayed Square Wave $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DS[:nf:ntimes]))
axs[2].set_title('Phase Spectrum of a Delayed Square Wave $∠Y[k]$')
#axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticks([0,1,2,3,4,5])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_yticks([-np.pi,0,np.pi])
axs[2].set_yticklabels(['-π','0','π'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/delayedsquarewave.png')

#########################################################3
# Differenced Square wave
dfs = np.zeros(s.shape)
dfs[1:] = np.diff(s)
DFS = np.fft.fft(dfs)
axs[0].clear()
axs[0].stem(dfs)
axs[0].set_title('Differenced Square Wave, $y[n]=x[n]-x[n-1]$')
axs[1].clear()
axs[1].stem(np.absolute(DFS[:nf:ntimes])/ntimes)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of a Differenced Square Wave, $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DFS[:nf:ntimes]))
axs[2].set_title('Phase Spectrum of a Differenced Square Wave, $∠Y[k]$')
#axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticks([0,1,2,3,4,5])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_yticks([-np.pi,0,np.pi])
axs[2].set_yticklabels(['-π','0','π'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/differenced_squarewave.png')

#########################################################3
# Delayed Differenced Square wave
ddfs = np.roll(dfs,5)
DDFS = np.fft.fft(ddfs)
axs[0].clear()
axs[0].stem(ddfs)
axs[0].set_title('Delayed Differenced Square Wave, $y[n]=x[n-5]-x[n-6]$')
axs[1].clear()
axs[1].stem(np.absolute(DDFS[:nf:ntimes])/ntimes)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of the DDS, $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DDFS[:nf:ntimes]))
axs[2].set_title('Phase Spectrum of the DDS, $∠Y[k]$')
#axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticks([0,1,2,3,4,5])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_yticks([-np.pi,0,np.pi])
axs[2].set_yticklabels(['-π','0','π'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/delayed_differenced_squarewave.png')
fig.savefig('exp/differenced_delayed_squarewave.png')

#########################################################3
# Local average filters: time domain
n = np.arange(-15,16)
fc = np.concatenate((np.zeros(10),np.ones(11)/11,np.zeros(10)))
fd = np.concatenate((np.zeros(15),np.ones(11)/11,np.zeros(5)))
fig.clf()
axs = fig.subplots(2,1)
axs[0].stem(n,fc)
axs[0].set_title('Impulse response $f_c[n]$ of a centered local averaging filter')
axs[1].stem(n,fd)
axs[1].set_title('Impulse response $f_d[n]$ of a delayed local averaging filter')
fig.tight_layout()
fig.savefig('exp/localaveragefilters.png')

#########################################################3
# Dirichlet form
dirichlet = np.concatenate((np.zeros(10),np.ones(11),np.zeros(10)))
axs[0].clear()
axs[0].stem(n,dirichlet)
axs[0].set_title('Centered local sum filter (Dirichlet form), $d_{11}[n]$')
omega = np.linspace(0,2*np.pi,ntimes*11)
D11 = 11*np.ones(omega.shape)
D11[1:] = np.sin(omega[1:]*11/2)/np.sin(omega[1:]/2)
axs[1].clear()
axs[1].set_title('Dirichlet Form $D_{11}(\omega)$')
axs[1].plot(np.arange(nf),D11[:nf],'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[1].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[1].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/dirichletform.png')

#########################################################3
# Centered local averaging filter
axs[0].clear()
axs[0].stem(n,fc)
axs[0].set_title('Centered local average filter, $f_c[n]$')
axs[1].clear()
axs[1].plot(np.arange(nf),D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_title('Magnitude Response of centered local averager, $F_c(\omega)$')
axs[1].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[1].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[1].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/centeredaveragingfilter.png')

#############################################################################
# Square wave plot
t=np.linspace(-1.5,1.5,3001,endpoint=True)
x = np.zeros(t.shape)
x[(-1.25<t)&(t<-0.75)]=1
x[(-0.25<t)&(t<0.25)]=1
x[(0.75<t)&(t<1.25)]=1
xticks = [-1.5,-1,-0.5,0,0.5,1,1.5]
xtick_labels = ['-1.5T_0','-T_0','-T_0/2','0','T_0/2','T_0','1.5T_0']
xax = np.zeros(t.shape)
yax = [np.array([-1e-6,1e-6]), np.array([-1.05,1.65])]
accum = np.zeros(len(t))
lasttitle = []
fig = matplotlib.figure.Figure(figsize=(6, 6))
axs = fig.subplots(1,1)
axs.plot(t,x)
axs.plot(t,x,'b-',t,xax,'k--',yax[0],yax[1],'k--')
axs.set_xticks(xticks)
axs.set_xticklabels(xtick_labels)
axs.set_title('Square wave example')
axs.set_xlabel('Time (seconds)')
fig.savefig('exp/squarewave_alone.png')


