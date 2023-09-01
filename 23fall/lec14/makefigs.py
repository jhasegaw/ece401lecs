import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure, subprocess, os, wave

os.makedirs('exp',exist_ok=True)

################################################################################
# Probably globally useful stuff
f_over = 256 # oversampling rate, in kHz
fs = 16 # sampling rate, in kHz
segdur = 1 # duration of t_axis, in ms
t_axis = np.arange(f_over*segdur)/f_over
n_axis = np.arange(fs*segdur)
def plot_axlines(ax, xlim, ylim):
    ax.plot([np.amin(xlim),np.amax(xlim)],[0,0],'k-',[0,1e-6],1.3*np.array([np.amin(ylim),np.amax(ylim)]),'k-')
    
def plot_sampled(ax0, ax1, t_axis, x_t, n_axis, x_n):
    ax0.plot(t_axis, x_t)
    ax0.stem(n_axis/fs, x_n, use_line_collection=True)
    plot_axlines(ax0,t_axis, x_t)
    ax0.set_xlabel('Time (ms)')
    ax1.stem(n_axis, x_n, use_line_collection=True)
    ax1.set_xlabel('Time (samples)')

def d2c(x_n, t_axis, pulse, fs, pulsehalfwidth):
    u = np.zeros(len(t_axis))
    for (n,x) in enumerate(x_n):
        u[t_axis==n/fs] = x
    y = np.convolve(u,pulse,mode='same')
    return(y)

def sinusoid(frequency, phasor, duration, samplerate):
    '''
    timeaxis, signal = sinusoid(frequency, phasor, duration, samplerate)
    Generate a sinusoid.

    frequency (real scalar) - frequency of the sinusoid, in Hertz
    phasor (complex scalar) - magnitude times e^{j phase}
    duration (real scalar) - duration, in seconds
    samplerate (real scalar) - sampling rate, in samples/second
    timeaxis (array) - sample times, from 0 to duration, including endpoints
    signal (array) - the generated sinusoid, length = int(duration*samplerate+1)
    '''
    #raise RuntimeError("You need to write this part!")
    #t = np.arange(int(duration*samplerate)+1)/samplerate
    t = np.linspace(0,duration,int(duration*samplerate)+1)
    return t, np.real(phasor*np.exp(1j*2*np.pi*frequency*t))

###########################################################################
# Picture, 3x2, showing time and frequency domain views of periodic CT signal, periodic with
# frequencies limited to +/-Nyquist, and corresponding periodic DT signal stem plot.
fk = [ np.arange(1,f,2) for f in [ f_over/4, fs/2, fs/2  ]]       
ak = [ np.array([ np.sin(k*np.pi/2)/(k*np.pi) for k in kset ]) for kset in fk ]
xt = [ np.zeros(t_axis.shape) for _ in range(3) ]
xn = [ np.zeros(n_axis.shape) for _ in range(3) ]
for row in range(3):
    for k in range(len(fk[row])):
        xt[row] += ak[row][k] * np.cos(4*np.pi*fk[row][k]*t_axis)
        xn[row] += ak[row][k] * np.cos(4*np.pi*fk[row][k]*n_axis/fs)

fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(3,2)
for row in range(2):
    axs[2,0].clear()
    plot_sampled(axs[row,0], axs[2,0], t_axis, xt[row], n_axis, xn[row])
    axs[row,0].set_title('$x(t)$ with frequencies up to %dkHz'%(np.amax(fk[row])))
    axs[2,0].set_title('$x[n]=x(n/16000)$'%(np.amax(fk[row])))
for row in range(3):
    axs[row,1].stem(fk[row],ak[row],use_line_collection=True)
    axs[row,1].stem(-fk[row],ak[row],use_line_collection=True)
    plot_axlines(axs[row,1], [-np.amax(fk[row])/4,np.amax(fk[row])/4], ak[row])
plot_axlines(axs[1,1],[-f_over/4,f_over/4],ak[1])
axs[0,1].set_title('Spectrum of $x(t)$ with frequencies up to +/-64 kHz')
axs[1,1].set_title('Spectrum of $x(t)$ with frequencies up to +/-8 kHz')
axs[2,1].set_title('Spectrum of $x[n]$')
axs[0,1].set_xlabel('Frequency (kHz)')
axs[1,1].set_xlabel('Frequency (kHz)')
axs[2,1].set_xlabel('Frequency (radians/sample)')
axs[2,1].set_xticks(np.concatenate((-fk[2],fk[2])))
axs[2,1].set_xticklabels(['-π/4','-π/2','-3π/4','-π','π/4','π/2','3π/4','π'])
fig.tight_layout()
fig.savefig('exp/periodic_nyquist.png')

###########################################################################
# Spectral plot of a sine at 800 Hz
fig, ax = plt.subplots(2,1,figsize=(7,8))
ax[0].plot(np.arange(41)/16000,np.sin(2*np.pi*800*np.arange(41)/16000))
ax[0].set_xlabel('t (sec)')
ax[0].set_title('$x(t)=\sin(2\pi 800t)$',fontsize=18)
ax[1].stem([-800,800],[0.5,0.5])
ax[1].plot([-1000,1000],[0,0],'k-',[0,1e-6],[0,0.75],'k-')
ax[1].annotate(text='$(1/2j)$',xy=(800,0.5),fontsize=18)
ax[1].annotate(text='$(-1/2j)$',xy=(-800,0.5),fontsize=18)
ax[1].set_title('Spectrum of $x(t)$',fontsize=18)
ax[1].set_xlabel('f (Hz)')
fig.tight_layout()
fig.savefig('exp/ct_sine.png')

# Spectral plot of 3x quadrature cosine at 800 Hz
fig, ax = plt.subplots(2,1,figsize=(7,8))
ax[0].plot(np.arange(41)/16000,3*np.cos(np.pi/4+2*np.pi*800*np.arange(41)/16000))
ax[0].set_xlabel('t (sec)')
ax[0].set_title('$x(t)=3\cos(2\pi 800t+\pi/4)$',fontsize=18)
ax[1].stem([-800,800],[1.5,1.5])
ax[1].plot([-1000,1000],[0,0],'k-',[0,1e-6],[0,2],'k-')
ax[1].annotate(text='$(3/2)e^{j\pi/4}$',xy=(800,1.5),fontsize=18)
ax[1].annotate(text='$(3/2)e^{-j\pi/4}$',xy=(-800,1.5),fontsize=18)
ax[1].set_title('Spectrum of $x(t)$',fontsize=18)
ax[1].set_xlabel('f (Hz)')
fig.tight_layout()
fig.savefig('exp/ct_quadrature.png')

###########################################################################
# Spectral plot of a sine at 800 Hz, sampled at 8000Hz
fig, ax = plt.subplots(2,1,figsize=(14,8))
ax[0].stem(np.arange(21),np.sin(2*np.pi*800*np.arange(21)/8000))
ax[0].set_xlabel('n (samples)')
ax[0].set_title('$x[n]=\sin(2\pi 800n/8000)=\sin(\pi n/5)$',fontsize=18)
ax[1].stem([-8800,-7200,-800,800,7200,8800],np.repeat(0.5,6))
ax[1].plot([-10000,10000],[0,0],'k-',[0,1e-6],[0,0.75],'k-')
for alias in [-8000,0,8000]:
    ax[1].annotate(text='$(1/2j)$',xy=(alias+800,0.5),fontsize=18)
    ax[1].annotate(text='$(-1/2j)$',xy=(alias-2600,0.5),fontsize=18)
for boundary in [-8000,-4000,4000,8000]:
    ax[1].plot([boundary,boundary+1e-6],[0,0.75],'k--')
ax[1].annotate(text='...',xy=(-10000,0.25),fontsize=24)
ax[1].annotate(text='...',xy=(10000,0.25),fontsize=24)
ax[1].set_xticks([-8000,-4000,-800,800,4000,8000])
ax[1].set_xticklabels(['$-2\pi$','$-\pi$','$-\omega$',
                    '$\omega$','$\pi$','$2\pi$'],fontsize=14)
ax[1].set_title('Spectrum of $x[n]$',fontsize=18)
fig.tight_layout()
fig.savefig('exp/dt_sine_oversampled.png')

# Spectral plot of a quadrature cosine at 800 Hz, sampled at 8000Hz
fig, ax = plt.subplots(2,1,figsize=(14,8))
ax[0].stem(np.arange(21),3*np.cos(np.pi/4+2*np.pi*800*np.arange(21)/8000))
ax[0].set_xlabel('n (samples)')
ax[0].set_title('$x(t)=3\cos(\pi/4+2\pi 800n/8000)=3\cos(\pi/4+\pi n/5)$',fontsize=18)
ax[1].stem([-8800,-7200,-800,800,7200,8800],np.repeat(1.5,6))
ax[1].plot([-10000,10000],[0,0],'k-',[0,1e-6],[0,2],'k-')
for alias in [-8000,0,8000]:
    ax[1].annotate(text='$(3/2)e^{j\pi/4}$',xy=(alias+800,1.5),fontsize=18)
    ax[1].annotate(text='$(3/2)e^{-j\pi/4}$',xy=(alias-2600,1.5),fontsize=18)
for boundary in [-8000,-4000,4000,8000]:
    ax[1].plot([boundary,boundary+1e-6],[0,2],'k--')
ax[1].annotate(text='...',xy=(-10000,0.25),fontsize=24)
ax[1].annotate(text='...',xy=(10000,0.25),fontsize=24)
ax[1].set_xticks([-8000,-4000,-800,800,4000,8000])
ax[1].set_xticklabels(['$-2\pi$','$-\pi$','$-\omega$',
                    '$\omega$','$\pi$','$2\pi$'],fontsize=14)
ax[1].set_title('Spectrum of $x[n]$',fontsize=18)
fig.tight_layout()
fig.savefig('exp/dt_quadrature_oversampled.png')

###########################################################################
# Spectral plot of a sine at 4800 Hz, sampled at 8000Hz
fig, ax = plt.subplots(2,1,figsize=(14,8))
ax[0].stem(np.arange(21),np.sin(2*np.pi*4800*np.arange(21)/8000))
ax[0].set_xlabel('n (samples)')
ax[0].set_title('$x[n]=\sin(2\pi 4800n/8000)=\sin(6\pi n/5)=-\sin(4\pi n/5)$',fontsize=18)
ax[1].stem([-4800,-3200,3200,4800],np.repeat(0.5,4))
ax[1].plot([-10000,10000],[0,0],'k-',[0,1e-6],[0,0.75],'k-')
ax[1].annotate(text='$(1/2j)$',xy=(4800,0.5),fontsize=18)
ax[1].annotate(text='$(1/2j)$',xy=(-3200,0.5),fontsize=18)
ax[1].annotate(text='$(-1/2j)$',xy=(-4800-1800,0.5),fontsize=18)
ax[1].annotate(text='$(-1/2j)$',xy=(3200-1800,0.5),fontsize=18)
for boundary in [-8000,-4000,4000,8000]:
    ax[1].plot([boundary,boundary+1e-6],[0,0.75],'k--')
ax[1].annotate(text='...',xy=(-10000,0.25),fontsize=24)
ax[1].annotate(text='...',xy=(10000,0.25),fontsize=24)
ax[1].set_xticks([-8000,-4800,-3200,3200,4800,8000])
ax[1].set_xticklabels(['$-2\pi$','$-\omega$','$-2\pi+\omega$',
                    '$2\pi-\omega$','$\omega$','$2\pi$'],fontsize=14)
ax[1].set_title('Spectrum of $x[n]$',fontsize=18)
fig.tight_layout()
fig.savefig('exp/dt_sine_undersampled.png')

# Spectral plot of a quadrature cosine at 800 Hz, sampled at 8000Hz
fig, ax = plt.subplots(2,1,figsize=(14,8))
ax[0].stem(np.arange(21),3*np.cos(np.pi/4+2*np.pi*4800*np.arange(21)/8000))
ax[0].set_xlabel('n (samples)')
ax[0].set_title('$x(t)=3\cos(\pi/4+2\pi 4800n/8000)=3\cos(\pi/4+6\pi n/5)=3\cos(-\pi/4+4\pi n/5)$',fontsize=18)
ax[1].stem([-4800,-3200,3200,4800],np.repeat(1.5,4))
ax[1].plot([-10000,10000],[0,0],'k-',[0,1e-6],[0,2],'k-')
ax[1].annotate(text='$(3/2)e^{j\pi/4}$',xy=(4800,1.5),fontsize=18)
ax[1].annotate(text='$(3/2)e^{j\pi/4}$',xy=(-3200,1.5),fontsize=18)
ax[1].annotate(text='$(3/2)e^{-j\pi/4}$',xy=(-4800-1800,1.5),fontsize=18)
ax[1].annotate(text='$(3/2)e^{-j\pi/4}$',xy=(3200-1800,1.5),fontsize=18)
for boundary in [-8000,-4000,4000,8000]:
    ax[1].plot([boundary,boundary+1e-6],[0,2],'k--')
ax[1].annotate(text='...',xy=(-10000,0.25),fontsize=24)
ax[1].annotate(text='...',xy=(10000,0.25),fontsize=24)
ax[1].set_xticks([-8000,-4800,-3200,3200,4800,8000])
ax[1].set_xticklabels(['$-2\pi$','$-\omega$','$-2\pi+\omega$',
                    '$2\pi-\omega$','$\omega$','$2\pi$'],fontsize=14)
ax[1].set_title('Spectrum of $x[n]$',fontsize=18)
fig.tight_layout()
fig.savefig('exp/dt_quadrature_undersampled.png')

################################################################################
# Probably globally useful stuff
f_over = 256 # oversampling rate, in kHz
fs = 16 # sampling rate, in kHz
segdur = 1 # duration of t_axis, in ms
t_axis = np.arange(f_over*segdur)/f_over
n_axis = np.arange(fs*segdur)
def plot_axlines(ax, xlim, ylim):
    ax.plot([np.amin(xlim),np.amax(xlim)],[0,0],'k-',[0,1e-6],1.3*np.array([np.amin(ylim),np.amax(ylim)]),'k-')
    
def plot_sampled(ax0, ax1, t_axis, x_t, n_axis, x_n):
    ax0.plot(t_axis, x_t)
    ax0.stem(n_axis/fs, x_n, use_line_collection=True)
    plot_axlines(ax0,t_axis, x_t)
    ax0.set_xlabel('Time (ms)')
    ax1.stem(n_axis, x_n, use_line_collection=True)
    ax1.set_xlabel('Time (samples)')

def d2c(x_n, t_axis, pulse, fs, pulsehalfwidth):
    u = np.zeros(len(t_axis))
    for (n,x) in enumerate(x_n):
        u[t_axis==n/fs] = x
    y = np.convolve(u,pulse,mode='same')
    return(y)

###########################################################################
# Picture showing a rectangular pulse, with X-axis labeled as -T_S/2, T_S/2.
fig = matplotlib.figure.Figure((14,6))
ax = fig.subplots()
t_pulse = np.arange(int(14*(f_over/fs))+1)/(f_over/fs) - 7
pulse = np.zeros(t_pulse.shape)
pulse[(t_pulse>=-0.5)&(t_pulse<0.5)] = 1
ax.plot(t_pulse, pulse)
plot_axlines(ax,t_pulse, pulse)
ax.set_title('Rectangular Pulse')
ax.set_xticks([-3,-2,-1,0,1,2,3])
ax.set_xticklabels(['-3Ts','-2Ts','-Ts','0','Ts','2Ts','3Ts'])
fig.savefig('exp/pulse_rectangular.png')

###########################################################################
# Picture showing x[n], and showing x(t) interpolated by rectangular pulses.
x_t = np.sin(2*2*np.pi*t_axis)
x_n = np.sin(2*2*np.pi*n_axis/fs)
y_t = d2c(x_n, t_axis, pulse, fs, 3.5)
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
plot_sampled(axs[1],axs[0],t_axis,y_t,n_axis,x_n)
axs[0].set_title('Discrete-time signal $x[n]$')
axs[1].set_title('$x[n]$ interpolated using a rectangular pulse')
fig.tight_layout()
fig.savefig('exp/interpolated_rectangular.png')

###########################################################################
# Picture showing a triangular pulse, with X-axis labeled as -T_s, T_s.
fig = matplotlib.figure.Figure((14,6))
ax = fig.subplots()
pulse = np.maximum(0,1-np.abs(t_pulse))
ax.plot(t_pulse, pulse)
plot_axlines(ax,t_pulse, pulse)
ax.set_title('Triangular Pulse')
ax.set_xticks([-3,-2,-1,0,1,2,3])
ax.set_xticklabels(['-3Ts','-2Ts','-Ts','0','Ts','2Ts','3Ts'])
fig.savefig('exp/pulse_triangular.png')

###########################################################################
# Picture showing x[n], and showing x(t) interpolated by triangular pulses.
y_t = d2c(x_n, t_axis, pulse, fs, 3.5)
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
plot_sampled(axs[1],axs[0],t_axis,y_t,n_axis,x_n)
axs[0].set_title('Discrete-time signal $x[n]$')
axs[1].set_title('$x[n]$ interpolated using a triangular pulse')
fig.tight_layout()
fig.savefig('exp/interpolated_triangular.png')

###########################################################################
#  Picture showing a cubic spline pulse, with X-axis labeled as -2T_S, -T_s, 0, T_s, 2T_s.
#p(t) = 1-\left(\frac{|t|}{T_S}\right)^2 & -T_S\le t<T_S\\
#    -2\left(\frac{|t|}{T_S}-1\right)\left(\frac{|t|}{T_S}-2\right)^2 & T_S\le |t|<2T_S\\
fig = matplotlib.figure.Figure((14,6))
ax = fig.subplots()
t = np.abs(t_pulse)
w = np.where(t<=1)
pulse[w]=1-1.5*np.square(t[w])+0.5*np.power(t[w],3)
w = np.where((t>=1)&(t<=2))
pulse[w] = -1.5*np.square(t[w]-2)*(t[w]-1)
ax.plot(t_pulse, pulse)
plot_axlines(ax,t_pulse, pulse)
ax.set_title('Cubic Spline Pulse')
ax.set_xticks([-3,-2,-1,0,1,2,3])
ax.set_xticklabels(['-3Ts','-2Ts','-Ts','0','Ts','2Ts','3Ts'])
fig.savefig('exp/pulse_spline.png')

###########################################################################
# Picture showing x[n], and showing x(t) interpolated by cubic pulses.
y_t = d2c(x_n, t_axis, pulse, fs, 3.5)
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
plot_sampled(axs[1],axs[0],t_axis,y_t,n_axis,x_n)
axs[0].set_title('Discrete-time signal $x[n]$')
axs[1].set_title('$x[n]$ interpolated using a cubic spline pulse')
fig.tight_layout()
fig.savefig('exp/interpolated_spline.png')

###########################################################################
# Picture showing a sinc pulse, with X-axis labeled as -2T_S, -T_s, 0, T_s, 2T_s.
fig = matplotlib.figure.Figure((14,6))
ax = fig.subplots()
pulse[t_pulse!=0] = np.sin(np.pi*t_pulse[t_pulse!=0])/(np.pi*t_pulse[t_pulse!=0])
pulse[t_pulse==0] = 1
ax.plot(t_pulse, pulse)
plot_axlines(ax,t_pulse, pulse)
ax.set_title('Sinc Pulse, $p(t)=sin(nT_s\pi)/nT_s\pi$')
ax.set_xticks([-3,-2,-1,0,1,2,3])
ax.set_xticklabels(['-3Ts','-2Ts','-Ts','0','Ts','2Ts','3Ts'])
fig.savefig('exp/pulse_sinc.png')

###########################################################################
# Picture showing x[n], and showing x(t) interpolated by sinc pulses.
y_t = d2c(x_n, t_axis, pulse, fs, fs)
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
plot_sampled(axs[1],axs[0],t_axis,y_t,n_axis,x_n)
axs[0].set_title('Discrete-time signal $x[n]$')
axs[1].set_title('$x[n]$ interpolated using a sinc pulse')
fig.tight_layout()
fig.savefig('exp/interpolated_sinc.png')

