import numpy as np
import matplotlib.figure, subprocess, os, wave
import matplotlib.pyplot as plt

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
# Picture with two axes showing sin at 1kHz sampled at 16kHz, then just  the stem plot.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
x_t = np.sin(2*np.pi*t_axis)
x_n = np.sin(2*np.pi*n_axis/fs)
plot_sampled(axs[0], axs[1], t_axis, x_t, n_axis, x_n)
axs[0].set_title('Continuous-time signal $x(t)=sin(2\pi 1000 t)$')
axs[1].set_title('Discrete-time signal $x[n]=sin(2\pi 1000 n/16000)=sin(\pi n/8)$')
fig.tight_layout()
fig.savefig('exp/sampled_sine1.png')

###########################################################################
# Picture with 2x2 axes showing cosines at 10kHz and 6kHz, sampled at 16kHz above, stem plots below.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,2)
f1 = 10
x1t = np.cos(2*np.pi*f1*t_axis)
x1n = np.cos(2*np.pi*f1*n_axis/fs)
plot_sampled(axs[0,0], axs[1,0], t_axis, x1t, n_axis, x1n)
axs[0,0].set_title('Continuous-time signal $x(t)=cos(2\pi 10000 t)$')
axs[1,0].set_title('Discrete-time signal $x[n]=cos(2\pi 10000 n/16000)=cos(5\pi n/4)=cos(3\pi n/4)$')
f2 = 6
x2t = np.cos(2*np.pi*f2*t_axis)
x2n = np.cos(2*np.pi*f2*n_axis/fs)
plot_sampled(axs[0,1], axs[1,1], t_axis, x2t, n_axis, x2n)
axs[0,1].set_title('Continuous-time signal $x(t)=cos(2\pi 6000 t)$')
axs[1,1].set_title('Discrete-time signal $x[n]=cos(2\pi 6000 n/16000)=cos(3\pi n/4)=cos(5\pi n/4)$')
fig.tight_layout()
fig.savefig('exp/sampled_aliasing.png')

###########################################################################
# Picture showing samples of an 8kHz cosine sampled at 16kHz.
fig = matplotlib.figure.Figure((14,6))
axs = fig.subplots(2,1)
f1 = 8
x1t = np.cos(2*np.pi*f1*t_axis)
x1n = np.cos(2*np.pi*f1*n_axis/fs)
plot_sampled(axs[0], axs[1], t_axis, x1t, n_axis, x1n)
axs[0].set_title('Continuous-time signal $x(t)=cos(2\pi 8000 t)$')
axs[1].set_title('Discrete-time signal $x[n]=cos(2\pi 8000 n/16000)=cos(\pi n)=(-1)^n$')
fig.tight_layout()
fig.savefig('exp/sampled_nyquist.png')
  
###########################################################################
# Pictures showing samples of 4Hz cosine sampled at different rates
fig, ax = plt.subplots(5,1, figsize=(14,10))
samplerates = [9, 8, 7, 6, 5]
for k in range(5):
    timeaxis, signal = sinusoid(4, 1, 1, samplerates[k])
    ax[k].stem(timeaxis, signal)
    alias = min(4, samplerates[k]-4)
    ax[k].set_title('4Hz, at $F_s$=%d Hz, looks like %dHz'%(samplerates[k], alias))
    ax[k].set_ylim(-1,1)
fig.tight_layout()
fig.savefig('exp/sampled_cosine1.png')
fig, ax = plt.subplots(5,1, figsize=(14,10))
samplerates = [4.5, 4, 3.5, 3, 2.5]
for k in range(5):
    timeaxis, signal = sinusoid(4, 1, 2, samplerates[k])
    ax[k].stem(timeaxis, signal)
    alias = min(4 % samplerates[k], (samplerates[k]-4) % samplerates[k])
    ax[k].set_title('4Hz, at $F_s$=%g Hz, looks like %gHz'%(samplerates[k], alias))
    ax[k].set_ylim(-1,1)
fig.tight_layout()
fig.savefig('exp/sampled_cosine2.png')

###########################################################################
# Pictures showing samples of 4Hz sine sampled at different rates
fig, ax = plt.subplots(5,1, figsize=(14,10))
samplerates = [10,9,8,7,6]
for k in range(5):
    timeaxis, signal = sinusoid(4, -1j, 1, samplerates[k])
    ax[k].stem(timeaxis, signal)
    alias = min(4 % samplerates[k], (samplerates[k]-4) % samplerates[k])
    ax[k].set_title('4Hz sine, at $F_s$=%g Hz, looks like %gHz'%(samplerates[k], alias))
    ax[k].set_ylim(-1,1)
fig.tight_layout()
fig.savefig('exp/sampled_sine2.png')

###########################################################################
# Pictures showing samples of 4Hz sinusoid at -pi/4 phase
fig, ax = plt.subplots(5,1, figsize=(14,10))
samplerates = [30,11,10,6,5]
phasor = np.exp(-1j*np.pi/4)
t_ref, x_ref = sinusoid(4, phasor, 1, 2*samplerates[0])
for k in range(5):
    phasor = np.exp(-1j*np.pi/4)
    timeaxis, signal = sinusoid(4, phasor, 1, samplerates[k])
    ax[k].stem(timeaxis, signal)
    alias = min(4 % samplerates[k], (samplerates[k]-4) % samplerates[k])
    if k <= 2:
        alias_phasor = '$-\pi/4$'
        ax[k].plot(t_ref, np.real(phasor*np.exp(1j*2*np.pi*alias*t_ref)),'k--')
    else:
        alias_phasor = '$+\pi/4$'        
        ax[k].plot(t_ref, np.real(np.conj(phasor)*np.exp(1j*2*np.pi*alias*t_ref)),'k--')
    ax[k].set_title('4Hz at $-\pi/4$ phase, at $F_s$=%g Hz, looks like %gHz with phase of %s'%(samplerates[k], alias, alias_phasor))
    ax[k].set_ylim(-1,1)
fig.tight_layout()
fig.savefig('exp/sampled_quarter.png')

###########################################################################
# Aliasing on a cosine and on a sine
fig, ax = plt.subplots(1,1,figsize=(14,4))
phi = np.linspace(0,2,200)
ax.plot(phi,np.cos(2*np.pi*phi),'b-',phi,np.zeros(len(phi)), 'k--')
xticks = [ 0, 0.3, 0.7, 1, 1.3 ]
for x in [0.3,0.7,1.3]:
    ax.plot([x-0.00001, x], [-1,1],'k--')
ax.set_xticks(xticks)
xticklabels = [ '0', '$\phi$', '$2\pi-\phi$', '$2\pi$', '$\phi+2\pi$' ]
ax.set_xticklabels(xticklabels, fontsize=18)
fig.savefig('exp/cosine_aliasing.png')
fig, ax = plt.subplots(1,1,figsize=(14,4))
phi = np.linspace(0,2,200)
ax.plot(phi,np.sin(2*np.pi*phi),'b-',phi,np.zeros(len(phi)), 'k--')
xticks = [ 0, 0.4, 0.6, 1, 1.4 ]
for x in [0.4,0.6,1.4]:
    ax.plot([x-0.00001, x], [-1,1],'k--')
ax.set_xticks(xticks)
xticklabels = [ '0', '$\phi$', '$2\pi-\phi$', '$2\pi$', '$\phi+2\pi$' ]
ax.set_xticklabels(xticklabels, fontsize=18)
fig.savefig('exp/sine_aliasing.png')
