import numpy as np
import matplotlib.figure, subprocess, os, wave

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
    axs[0].stem(nset,x,use_line_collection=True)
    ylim = [ min(-0.1,1.1*np.amin(x)), max(1.1,1.1*np.amax(x))  ]
    axs[0].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[0].set_title('$x[m]=\delta[m]$')
    axs[1].clear()
    hplot = np.zeros(nset.shape)
    hplot[nset <= n] = h[n-nset[nset<=n]]
    axs[1].stem(nset,hplot,use_line_collection=True)
    ylim = [ min(-1.1,1.1*np.amin(h)), max(1.1,1.1*np.amax(h))  ]
    axs[1].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[1].set_title('$h[%d-m]$'%(n))
    axs[2].clear()
    y[nset==n] = np.sum(hplot[nset%1==0]*x[nset%1==0])
    axs[2].stem(nset,y,use_line_collection=True)
    axs[2].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],1.1*np.array(yylim),'k-')
    axs[2].set_title('$y[m]=h[m]*x[m]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    return(y)

###########################################################################
# Video showing the frequency response  of damped resonator with a finite peak.
N = 50
omega1 = np.pi/5
sigma1 = 0.1
nset = np.arange(-int(0.2*N),N+1)
x = np.zeros(nset.shape)
x[nset==0]=1
h = np.exp(-sigma1*(nset-np.amin(nset)))*np.sin(omega1*(1+nset-np.amin(nset)))/np.sin(omega1)
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]

fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
a = [np.exp(-0.1+2j*np.pi/5), np.exp(-0.1-2j*np.pi/5)]
H = np.ones(omega.shape)
for k in range(len(a)):
    H *= 1/np.abs(1-a[k]*np.exp(-1j*omega))
for m in range(50):
    n=100*m
    axs[0].clear()
    plot_circle(axs[0], pole=a, ptext=['$p_1$','$p_2$'])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    for k in range(len(a)):
        axs[0].plot([np.real(a[k]),np.cos(omega[n])],[np.imag(a[k]),np.sin(omega[n])],'b-')
    axs[0].set_title('Pole-Zero Plot, $p_1=exp(-0.1+j2π/5)$')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'m-')
    axs[1].set_ylim([0,6])
    axs[1].set_title('$|H(\omega)|$, max≈$1/0.1=10$, bandwidth=0.2')
    fig.savefig('exp/dampedfreq%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/dampedfreq?.png exp/dampedfreq??.png exp/dampedfreq.gif'.split())

###########################################################################
# Video showing the convolution of cosine input with damped resonator, going toward maximum value.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
omega1 = np.pi/5
nset = np.arange(-int(0.2*N),N+1)
x = np.zeros(nset.shape)
x[nset>=0]=np.cos(omega1*nset[nset>=0])
h = np.exp(-sigma1*(nset-np.amin(nset)))*np.sin(omega1*(1+nset-np.amin(nset)))/np.sin(omega1)
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/dampedconv%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/dampedconv?.png exp/dampedconv??.png exp/dampedconv.gif'.split())

###########################################################################
# Picture of speech signals, showing damped sinusoids.
with wave.open('ow.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    ow = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    ow /= np.amax(np.abs(ow))
fig = matplotlib.figure.Figure((10,4))
ax = fig.subplots()
t = np.arange(len(ow))/16000
ax.plot(t,ow)
ax.set_title('Waveform of the vowel /o/')
ax.set_xlabel('Time (sec)')
fig.savefig('exp/speechwave.png')
    

################################################################################
# Picture  showing glottal input, filter, speech output.
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)
N = 400
n0 = [ 45+80*m for m in range(5) ]
G = 7.2
delta = np.zeros(N)
delta[0] = 1
F = np.array([800, 1200, 2800, 3200])
B = np.array([100, 100, 400, 600])
fs = 8000
p = np.exp(-np.pi*B/fs)*np.exp(2j*np.pi*F/fs)
h = delta.copy()
for k in range(4):
    h[1] = h[1] + 2*np.real(p[k])*h[0]
    for n in range(2,N):
        h[n] = h[n] + 2*np.real(p[k])*h[n-1] - np.square(np.abs(p[k]))*h[n-2]
x = np.zeros(N)
y = np.zeros(N)
for m in range(5):
    x -= G*np.concatenate((delta[N-n0[m]:N],delta[:N-n0[m]]))
    y -= G*np.concatenate((h[N-n0[m]:N],h[:N-n0[m]]))
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Air pressure at glottis = series of negative impulses')
axs[1].plot(h)
axs[1].set_title('Impulse response of the vocal tract = damped resonances')
axs[2].plot(y)
axs[2].set_title('Air pressure at lips = series of damped resonances')
axs[2].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speech_fivepulses.png')

################################################################################
# Picture showing impulse input, speech output.
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)
N = 80
n0 = 45
delta = np.zeros(N)
delta[0] = 1
h = delta.copy()
for k in range(4):
    h[1] = h[1] + 2*np.real(p[k])*h[0]
    for n in range(2,N):
        h[n] = h[n] + 2*np.real(p[k])*h[n-1] - np.square(np.abs(p[k]))*h[n-2]
x = -G*np.concatenate((delta[N-n0:N],delta[:N-n0]))
y = -G*np.concatenate((h[N-n0:N],h[:N-n0]))
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Air pressure at glottis = $G\delta[n-n_0]$, once per frame')
axs[1].plot(h)
axs[1].set_title('Impulse response of the vocal tract')
axs[2].plot(y)
axs[2].set_title('Air pressure at lips = $Gh[n-n_0]$, once per frame')
axs[2].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speech_onepulse.png')

################################################################################
# Picture showing the inverse filtering result, maybe from a real speech signal.
with wave.open('ow.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    ow = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    ow /= np.amax(np.abs(ow))

r = np.correlate(ow,ow,'full')
print(r[398:403])
print(len(r))
r = r[int((len(r)-1)/2):]
print(len(r))
R = np.diag(np.tile(r[0],10))
for m in range(1,10):
    R += np.diag(np.tile(r[m],10-m),m)
    R += np.diag(np.tile(r[m],10-m),-m)
gamma = r[1:11]
a = np.matmul(np.linalg.inv(R),gamma.reshape((10,1)))
e = np.zeros(ow.shape)
for n in range(len(ow)):
    e[n] = ow[n]
    for m in range(min(n,10)):
        e[n] -= a[m]*ow[n-(m+1)]
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,1)
t = np.arange(len(ow))/16000
axs[0].plot(t,ow)
axs[0].set_title('Waveform, $s[n]$, of the vowel /o/')
axs[1].stem(np.arange(1,11),a,use_line_collection=True)
axs[1].set_title('Predictor Coefficients $a_k$')
axs[2].plot(t[11:],e[11:])
axs[2].set_title('Result of Inverse Filtering, $e[n]=s[n]-sum_k a_k s[n-k]$')
axs[2].set_xlabel('Time (sec)')
fig.tight_layout()
fig.savefig('exp/inversefilter.png')
