import numpy as np
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
fig.savefig('exp/sampled_sine.png')

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
w = np.where(t<1)
pulse[w]=1-2*np.square(t[w])+np.power(t[w],3)
w = np.where((t>1)&(t<2))
pulse[w] = -np.square(2-t[w])+np.power(2-t[w],3)
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

