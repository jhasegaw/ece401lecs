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
fig.savefig('phaseshift.png')

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
fig.savefig('startingpoints.png')

# triple-beat figure
fig = matplotlib.figure.Figure(figsize=(5, 4))
axs = fig.subplots(4,1,sharex=True)
fig.tight_layout()
t = np.linspace(0,0.5,4000,endpoint=False)
f = [107,110,104]
x = []
for k in range(3):
    x.append(np.cos(2*np.pi*f[k]*t))
    axs[k].plot(t,x[k])
    axs[k].set_title('cos(2π%dt)'%(f[k]))
axs[3].plot(t,x[0]+x[1]+x[2])
axs[3].set_title('Sum = average tone, beating at 1+2cos(2π3t)')
axs[3].set_xlabel('Time (t)')
fig.savefig('triplebeat.png')

ω = 2
