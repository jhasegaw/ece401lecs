import numpy as np
import matplotlib.figure

# phase-shift figure
fig = matplotlib.figure.Figure(figsize=(4, 4))
axs = fig.subplots(5,1,sharex=True)
fig.tight_layout()
t = np.linspace(-2*np.pi,2*np.pi,1000,endpoint=True)
phi = [-2, -1, 0, 1, 2]
phase = ['-2π/3','-π/3','','+π/3','+2π/3']
for k in range(5):
    axs[k].plot(t,np.cos(t+phi[k]*np.pi/3),'b-',[-1e-6,1e-6],[-1,1],'r--')
    axs[k].set_title('cos(t%s)'%(phase[k]))
axs[4].set_xlabel('Time (t)')
fig.savefig('exp/phaseshift.png')

# phase-shift positions figure
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.set_aspect('equal')
t = np.linspace(-np.pi,np.pi,1000,endpoint=True)
x = np.cos(t)
y = np.sin(t)
ax.plot(x,y,'b--',[-1.5,1.5],[-1e-6,1e-6],'k--',[-1e-6,1e-6],[-1.5,1.5],'k--')
phi = [-2, -1, 0, 1, 2]
phase = ['θ=-2π/3','θ=-π/3','θ=0','θ=π/3','θ=2π/3']
for k in range(5):
    ax.plot(np.cos(np.pi*phi[k]/3),np.sin(np.pi*phi[k]/3),'rD')
    ax.text(1.25*np.cos(np.pi*phi[k]/3)-0.2,1.25*np.sin(np.pi*phi[k]/3),phase[k])
fig.savefig('exp/startingpoints.png')

