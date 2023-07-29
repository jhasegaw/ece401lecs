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

