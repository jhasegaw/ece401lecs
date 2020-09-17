import numpy as np
import math,subprocess
import matplotlib.figure
import matplotlib.pyplot

squarewave = np.array([1,1,1,-0.5,-1,-1,-1,-1,-0.5,1,1])
ntimes = 5
s = np.tile(squarewave,ntimes)
S = np.fft.fft(s)
nf = int(len(S)/2)+1
f = matplotlib.figure.Figure(figsize=(5, 4))
a = f.subplots(2,1)
a[0].stem(s,use_line_collection=True)
a[0].set_title('Square Wave of length 11, $x[n]$')
a[0].set_xticks([0,11,22,33,44])
a[1].clear()
a[1].stem(np.real(S[:nf:ntimes])/ntimes,use_line_collection=True)
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
axs[0].stem(g,use_line_collection=True)
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
squarewave = np.array([1,1,1,-0.5,-1,-1,-1,-1,-0.5,1,1])
ntimes = 5
s = np.tile(squarewave,ntimes)
S = np.fft.fft(s)
k = np.linspace(0,len(squarewave),len(S),endpoint=False)
nf = int(len(S)/2)+1
axs[0].clear()
axs[0].stem(s,use_line_collection=True)
axs[0].set_title('Square Wave of length 11, $x[n]$')
axs[0].set_xticks([0,11,22,33,44])
axs[1].clear()
axs[1].stem(np.absolute(S[:nf:ntimes])/ntimes,use_line_collection=True)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of the Square Wave, $|X[k]|$')
axs[2].clear()
axs[2].stem(np.angle(S[:nf:ntimes]),use_line_collection=True)
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
axs[0].stem(ds,use_line_collection=True)
axs[0].set_title('Delayed Square Wave of length 11, $y[n]=x[n-5]$')
axs[0].set_xticks([0,11,22,33,44])
axs[1].clear()
axs[1].stem(np.absolute(DS[:nf:ntimes])/ntimes,use_line_collection=True)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of a Delayed Square Wave $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DS[:nf:ntimes]),use_line_collection=True)
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
axs[0].stem(dfs,use_line_collection=True)
axs[0].set_title('Differenced Square Wave, $y[n]=x[n]-x[n-1]$')
axs[1].clear()
axs[1].stem(np.absolute(DFS[:nf:ntimes])/ntimes,use_line_collection=True)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of a Differenced Square Wave, $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DFS[:nf:ntimes]),use_line_collection=True)
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
axs[0].stem(ddfs,use_line_collection=True)
axs[0].set_title('Delayed Differenced Square Wave, $y[n]=x[n-3]-x[n-4]$')
axs[1].clear()
axs[1].stem(np.absolute(DDFS[:nf:ntimes])/ntimes,use_line_collection=True)
axs[1].set_xticks([])
axs[1].set_title('Magnitude Spectrum of the DDS, $|Y[k]|$')
axs[2].clear()
axs[2].stem(np.angle(DDFS[:nf:ntimes]),use_line_collection=True)
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
axs[0].stem(n,fc,use_line_collection=True)
axs[0].set_title('Impulse response $f_c[n]$ of a centered local averaging filter')
axs[1].stem(n,fd,use_line_collection=True)
axs[1].set_title('Impulse response $f_d[n]$ of a delayed local averaging filter')
fig.tight_layout()
fig.savefig('exp/localaveragefilters.png')

#########################################################3
# Dirichlet form
dirichlet = np.concatenate((np.zeros(10),np.ones(11),np.zeros(10)))
axs[0].clear()
axs[0].stem(n,dirichlet,use_line_collection=True)
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
axs[0].stem(n,fc,use_line_collection=True)
axs[0].set_title('Centered local average filter, $f_c[n]$')
axs[1].clear()
axs[1].plot(np.arange(nf),D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_title('Magnitude Response of centered local averager, $F_c(\omega)$')
axs[1].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[1].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[1].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/centeredaveragingfilter.png')

#########################################################3
# Video: local averaging of the square wave
fig.clf()
axs = fig.subplots(3,1,sharex=True)
axs[0].stem(s,use_line_collection=True)
axs[0].set_title('Square wave $x[m]$')
for n in range(len(s)):
    f = np.zeros(s.shape)
    if n < 5:
        f[:(n+6)] = 1.0/11
        f[(n-5):] = 1.0/11
    elif n+6 > len(s):
        f[(n-5):] = 1.0/11
        f[:(n+6-len(s))] = 1.0/11
    else:
        f[(n-5):(n+6)] = 1.0/11
    axs[1].clear()
    axs[1].stem(f,use_line_collection=True)
    axs[1].set_title('Centered Local Averaging Filter $h[n-m]$ for $n=%d$'%(n))
    axs[2].clear()
    axs[2].stem(np.zeros(n+1),use_line_collection=True)
    axs[2].set_xlim(-1,len(s))
    axs[2].set_ylim(-1,1)
    axs[2].set_title('Filter Output $y[n]$ for $n=%d$'%(n))
    fig.tight_layout()
    fig.savefig('exp/localaverage_periodic%d.png'%(n))
subprocess.call('convert -delay 20 -dispose previous exp/localaverage_periodic?.png exp/localaverage_periodic??.png  exp/localaverage_periodic.gif'.split())

#########################################################3
# Denoising impulse response
delta = np.zeros(len(fc))
n = np.arange(len(fc))
delta[15] = 1
denoising_filter = delta - fc
axs[0].clear()
axs[0].stem(n,delta,use_line_collection=True)
axs[0].set_title('Impulse (Delta Function), $\delta[n]$')
axs[0].set_ylim(-0.25,1.25)
axs[1].clear()
axs[1].stem(n,fc,use_line_collection=True)
axs[1].set_title('Centered local averaging filter, $f_c[n]$')
axs[1].set_ylim(-0.25,1.25)
axs[2].clear()
axs[2].stem(n,denoising_filter,use_line_collection=True)
axs[2].set_ylim(-0.25,1.25)
axs[2].set_title('Denoising filter $\delta[n]-f_c[n]$')
fig.tight_layout()
fig.savefig('exp/denoising_impulseresponse.png')

#########################################################3
# Denoising frequency response
axs[0].clear()
axs[0].plot(np.ones(nf))
axs[0].set_ylim(-0.25,1.25)
axs[0].set_title('Frequency Response of a Delta Function $=1$ for all $\omega$')
axs[1].clear()
axs[1].plot(np.arange(nf),D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_title('Frequency Response of Centered Local Averaging Filter $F_c(\omega)$')
axs[2].clear()
axs[2].plot(np.arange(nf),np.ones(nf)-D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[2].set_title('Frequency Response of Denoising Filter, $1-F_c(\omega)$')
axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/denoising_frequencyresponse.png')

#########################################################3
# Spectrum of local averaged periodic signal
axs[0].clear()
axs[0].stem(np.absolute(S[:nf])/ntimes,use_line_collection=True)
axs[0].set_title('Magnitude Spectrum of Square Wave $|X[k]|$')
axs[1].clear()
axs[1].plot(np.arange(nf),D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_title('Frequency Response of Local Averaging Filter, $F_c(\omega)$')
axs[2].clear()
Y = np.absolute(S)*D11/11
axs[2].plot(np.arange(nf),np.zeros(nf),'-',np.arange(nf),np.zeros(nf),'--')
axs[2].set_ylim(-0.25,1.25)
axs[2].set_title('Magnitude Spectrum of $|Y[k]|=|F_c(k\omega_0)X[k]|$')
axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/localaverage_spectrum.png')


#########################################################3
# Noisy signal
axs[0].clear()
axs[0].stem(s,use_line_collection=True)
axs[0].set_title('Signal')
axs[1].clear()
v = 0.1*np.random.normal(size=len(s))
axs[1].stem(v,use_line_collection=True)
axs[1].set_title('Noise')
axs[2].clear()
axs[2].stem(v+s,use_line_collection=True)
axs[2].set_title('Noisy Signal $x[n]$')
fig.tight_layout()
fig.savefig('exp/noisysignal.png')

#########################################################3
# Video of noising
x = v+s
axs[0].clear()
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Noisy Signal $x[m]$')
axs[0].set_ylim(-1.25,1.25)
y = np.array([])
noising_filter_center = -np.ones(11)/11
for n in range(len(s)):
    f = np.zeros(s.shape)
    if n < 5:
        f[:(n+6)] = noising_filter_center[-(n+6):]
        f[(n-5):] = noising_filter_center[:(5-n)]
    elif n+6 > len(s):
        f[(n-5):] = noising_filter_center[:(len(s)-(n-5))]
        f[:(n+6-len(s))] = noising_filter_center[(len(s)-(n+6)):]
    else:
        f[(n-5):(n+6)] = noising_filter_center
    axs[1].clear()
    axs[1].stem(f,use_line_collection=True)
    axs[1].set_title('Centered Local Averaging Filter $f_c[n-m]$ for $n=%d$'%(n))
    y=np.append(y,np.sum(f*x))
    axs[2].clear()
    axs[2].stem(y,use_line_collection=True)
    axs[2].set_xlim(-1,len(s))
    axs[2].set_ylim(-0.1,0.05)
    axs[2].set_title('Filter Output $y[n]=f_c[n]*x[n]$ for $n=%d$'%(n))
    fig.tight_layout()
    fig.savefig('exp/noised_signal%d.png'%(n))
subprocess.call('convert -delay 20 -dispose previous exp/noised_signal?.png exp/noised_signal??.png  exp/noised_signal.gif'.split())

#########################################################3
# Video of denoising
x = v+s
axs[0].clear()
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Noisy Signal $x[m]$')
axs[0].set_ylim(-1.25,1.25)
y = np.array([])
denoising_filter_center = -np.ones(11)/11
denoising_filter_center[5] += 1
for n in range(len(s)):
    f = np.zeros(s.shape)
    if n < 5:
        f[:(n+6)] = denoising_filter_center[-(n+6):]
        f[(n-5):] = denoising_filter_center[:(5-n)]
    elif n+6 > len(s):
        f[(n-5):] = denoising_filter_center[:(len(s)-(n-5))]
        f[:(n+6-len(s))] = denoising_filter_center[(len(s)-(n+6)):]
    else:
        f[(n-5):(n+6)] = denoising_filter_center
    axs[1].clear()
    axs[1].stem(f,use_line_collection=True)
    axs[1].set_title('Centered Denoising Filter $h[n-m]$ for $n=%d$'%(n))
    y=np.append(y,np.sum(f*x))
    axs[2].clear()
    axs[2].stem(y,use_line_collection=True)
    axs[2].set_xlim(-1,len(s))
    axs[2].set_ylim(-1.25,1.25)
    axs[2].set_title('Filter Output $y[n]=h[n]*x[n]$ for $n=%d$'%(n))
    fig.tight_layout()
    fig.savefig('exp/denoised_signal%d.png'%(n))
subprocess.call('convert -delay 20 -dispose previous exp/denoised_signal?.png exp/denoised_signal??.png  exp/denoised_signal.gif'.split())

#########################################################3
# Spectrum of denoised signal
axs[0].clear()
X = np.fft.fft(x)
axs[0].plot(np.arange(nf),np.absolute(X[:nf])/ntimes,'-',np.arange(nf),np.zeros(nf),'--')
axs[0].set_title('Magnitude Spectrum of Noisy Square Wave $|X[k]|$')
axs[1].clear()
axs[1].plot(np.arange(nf),np.ones(nf)-D11[:nf]/11,'-',np.arange(nf),np.zeros(nf),'--')
axs[1].set_title('Frequency Response of Denoising Filter, $1-F_c(\omega)$')
axs[2].clear()
Y = np.fft.fft(y)
axs[2].plot(np.arange(nf),np.absolute(Y[:nf])/ntimes,'-',np.arange(nf),np.zeros(nf),'--')
axs[2].set_title('Magnitude Spectrum of Denoised Square Wave $|Y[k]|$')
axs[2].set_xticks([0,ntimes,2*ntimes,3*ntimes,4*ntimes,5*ntimes])
axs[2].set_xticklabels(['0','2π/11','4π/11','6π/11','8π/11','10π/11'])
axs[2].set_xlabel('Frequency $\omega$ (radians/sample)')
fig.tight_layout()
fig.savefig('exp/denoised_spectrum.png')

