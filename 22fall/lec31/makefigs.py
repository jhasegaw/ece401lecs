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
# Picture showing $p_1$ on the unit circle.
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()
a = [np.exp(2j*np.pi/5), np.exp(-2j*np.pi/5)]
plot_circle(ax, pole=a, ptext=['$p_1$','$p_2$'])
fig.savefig('exp/resonatorpoles.png')

###########################################################################
# Video showing the convolution of delta input with ideal resonator, generating resonant output.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
omega1 = np.pi/5
nset = np.arange(-int(0.2*N),N+1)
x = np.zeros(nset.shape)
x[nset==0]=1
h = np.sin(omega1*(1+nset-np.amin(nset)))/np.sin(omega1)
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/resonatorimpulse%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/resonatorimpulse?.png exp/resonatorimpulse??.png exp/resonatorimpulse.gif'.split())

###########################################################################
# Video showing the frequency response  of ideal resonator with a peak at infinity.
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
a = [np.exp(2j*np.pi/5), np.exp(-2j*np.pi/5)]
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
    axs[0].set_title('Pole-Zero Plot, $p_1=exp(j2π/5)$')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'m-')
    axs[1].set_ylim([0,6])
    fig.savefig('exp/resonatorfreq%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/resonatorfreq?.png exp/resonatorfreq??.png exp/resonatorfreq.gif'.split())

###########################################################################
# Video showing the convolution of cosine input with ideal resonator, going toward infinity.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
N = 50
omega1 = np.pi/5
nset = np.arange(-int(0.2*N),N+1)
x = np.zeros(nset.shape)
x[nset>=0]=np.cos(omega1*nset[nset>=0])
h = np.sin(omega1*(1+nset-np.amin(nset)))/np.sin(omega1)
y = np.zeros(nset.shape)
yy = np.convolve(h,x)
yylim = [np.amin(yy), np.amax(yy)]
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/resonatorconv%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/resonatorconv?.png exp/resonatorconv??.png exp/resonatorconv.gif'.split())

###########################################################################
# Picture showing $p_1$ inside the unit circle.
fig = matplotlib.figure.Figure((6,4))
ax = fig.subplots()
a = [np.exp(-0.1+2j*np.pi/5), np.exp(-0.1-2j*np.pi/5)]
plot_circle(ax, pole=a, ptext=['$p_1$','$p_2$'])
fig.savefig('exp/dampedpoles.png')

###########################################################################
# Video showing the convolution of delta input with stable resonator, generating decaying sinusoid output.
fig = matplotlib.figure.Figure((5,4))
axs = fig.subplots(3,1,sharex=True)
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
for n in range(N+1):
    y  = plot_convolution(axs, x, h, y, nset, n, yylim)
    fig.savefig('exp/dampedimpulse%d.png'%(n))
subprocess.call('convert -delay 10 -dispose previous exp/dampedimpulse?.png exp/dampedimpulse??.png exp/dampedimpulse.gif'.split())

###########################################################################
# Video showing the frequency response  of damped resonator with a finite peak.
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
    
###########################################################################
# Plots showing waveforms of /i/ and /a/, time domains in milliseconds and in samples.
with wave.open('aa.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    aa = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    aa /= np.amax(np.abs(aa))
with wave.open('iy.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    iy = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    iy /= np.amax(np.abs(iy))
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
t = np.arange(len(aa))/16000
axs[0,0].plot(t,aa)
axs[0,0].set_title('Waveform of the vowel /a/, $F_1=1000$Hz')
axs[0,0].set_xlabel('Time (sec)')
axs[1,0].plot(t,iy)
axs[1,0].set_title('Waveform of the vowel /i/, $F_1=300$Hz')
axs[1,0].set_xlabel('Time (sec)')
axs[0,1].plot(aa)
axs[0,1].set_title('Waveform of the vowel /a/, $\omega_1=π/8$')
axs[0,1].set_xlabel('Time (samples)')
axs[1,1].plot(iy)
axs[1,1].set_title('Waveform of the vowel /i/, $\omega_1=π/24$')
axs[1,1].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speechwaves.png')

###########################################################################
# Plots showing spectra of /i/ and /a/, freq domains in Hertz and in radians/sample.
L = int(0.015*16000)
w = np.hamming(L)
print(np.amin(w),np.amax(w))
print(np.amin(aa),np.amax(aa))
N = 8192
AA = np.zeros(N)
IY = np.zeros(N)
for n in range(0,len(aa)-L,int(L/2)):
    AA += np.square(np.abs(np.fft.fft(aa[n:(n+L)]*w, n=N)))
    IY += np.square(np.abs(np.fft.fft(iy[n:(n+L)]*w, n=N)))
print(np.amin(AA),np.amax(AA))
AA = np.sqrt(AA)
IY = np.sqrt(IY)
print(np.amin(AA),np.amax(AA))
f = np.linspace(0,16000,N,endpoint=False)
omega = 2*np.pi*f/16000
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
axs[0,0].plot(f[f<3500],20*np.log10(AA[f<3500]))
axs[0,0].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /a/, $F_1=800$Hz')
axs[0,0].set_xlabel('Freq (Hz)')
axs[1,0].plot(f[f<3500],20*np.log10(IY[f<3500]))
axs[1,0].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /i/, $F_1=400$Hz')
axs[1,0].set_xlabel('Freq (Hz)')
axs[0,1].plot(omega[f<3500],20*np.log10(AA[f<3500]))
axs[0,1].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /a/, $\omega_1=π/10$')
axs[0,1].set_xlabel('Freq (radians/sample)')
axs[1,1].plot(omega[f<3500],20*np.log10(IY[f<3500]))
axs[1,1].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /i/, $\omega_1=π/20$')
axs[1,1].set_xlabel('Freq (radians/sample)')
fig.tight_layout()
fig.savefig('exp/speechspecs.png')


