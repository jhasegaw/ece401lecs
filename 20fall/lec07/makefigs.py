import numpy as np
import math,subprocess
import matplotlib.figure
import matplotlib.gridspec
from matplotlib.patches import FancyArrowPatch

omega = np.linspace(0,np.pi,100)
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.plot(omega,omega,'--',omega,2*np.sin(omega/2),'-')
ax.set_title('$|G(\omega)|$ (solid) $≈\omega$ (dashed)')
ax.set_xlabel('Frequency (radians/sample)')
ax.set_ylabel('Magnitude Response $|G(\omega)|$')
fig.savefig('exp/firstdiff_magnitude.png')

const = (np.pi/2)*np.ones(omega.shape)
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.plot(omega,const,'--',omega,const-omega/2,'-')
ax.set_title('$∠G(\omega)$ (solid) $≈\pi/2$ (dashed)')
ax.set_xlabel('Frequency (radians/sample)')
ax.set_ylabel('Phase Response $∠G(\omega)$')
fig.savefig('exp/firstdiff_phase.png')

n = np.linspace(0,50,501,endpoint=True)
fig = matplotlib.figure.Figure(figsize=(10, 4))
axs = fig.subplots(2,1,sharex=True)
for k in np.arange(151):
    omega = np.pi*k/150
    x = np.cos(omega*n)
    y = 2*np.sin(omega/2)*np.cos(omega*n+(np.pi-omega)/2)
    axs[0].clear()
    axs[0].plot(x)
    axs[0].set_ylim([-2.1,2.1])
    axs[0].set_title('$x[n]=cos(\omega n)$ and $y[n]=x[n]-x[n-1]$, $\omega=%3.3g$'%(omega))
    axs[0].set_ylabel('$x[n]$')
    axs[1].clear()
    axs[1].plot(y)
    axs[1].set_ylabel('$y[n]$')
    axs[1].set_ylim([-2.1,2.1])
    fig.savefig('exp/firstdiff_tonesweep%d.png'%(k))

subprocess.call('convert -delay 7 -dispose previous exp/firstdiff_tonesweep?.png exp/firstdiff_tonesweep??.png exp/firstdiff_tonesweep???.png  exp/firstdiff_tonesweep.gif'.split())
