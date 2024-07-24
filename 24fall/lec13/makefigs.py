import numpy as np
import math,subprocess,os
import matplotlib.figure
import matplotlib.gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

os.makedirs('exp',exist_ok=True)

###################################################################
omega = np.linspace(0,np.pi,100)
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.plot(omega,2*np.sin(omega/2),'-')
ax.set_title('$|G(\omega)|$')
ax.set_xlabel('Frequency (radians/sample)')
ax.set_ylabel('Magnitude Response $|G(\omega)|$')
fig.savefig('exp/firstdiffonly_magnitude.png')

###################################################################
omega = np.linspace(0,np.pi,100)
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.plot(omega,omega,'--',omega,2*np.sin(omega/2),'-')
ax.set_title('$|G(\omega)|$ (solid) $≈\omega$ (dashed)')
ax.set_xlabel('Frequency (radians/sample)')
ax.set_ylabel('Magnitude Response $|G(\omega)|$')
fig.savefig('exp/firstdiff_magnitude.png')

###################################################################
const = (np.pi/2)*np.ones(omega.shape)
fig = matplotlib.figure.Figure(figsize=(5, 4))
ax = fig.subplots()
ax.plot(omega,const,'--',omega,const-omega/2,'-')
ax.set_title('$∠G(\omega)$ (solid) $≈\pi/2$ (dashed)')
ax.set_xlabel('Frequency (radians/sample)')
ax.set_ylabel('Phase Response $∠G(\omega)$')
fig.savefig('exp/firstdiff_phase.png')

###################################################################
n = np.linspace(0,50,501,endpoint=True)
fig = matplotlib.figure.Figure(figsize=(10, 4))
axs = fig.subplots(2,1,sharex=True)
for k in [10,75,150]:
    omega = np.pi*k/150
    x = np.cos(omega*n)
    y = 2*np.sin(omega/2)*np.cos(omega*n+(np.pi-omega)/2)
    axs[0].clear()
    axs[0].plot(n,x)
    axs[0].set_ylim([-2.1,2.1])
    axs[0].set_title('$x[n]=cos(\omega n)$ and $y[n]=x[n]-x[n-1]$, $\omega=%3.3g$'%(omega))
    axs[0].set_ylabel('$x[n]$')
    axs[1].clear()
    axs[1].plot(n,y)
    axs[1].set_ylabel('$y[n]$')
    axs[1].set_ylim([-2.1,2.1])
    fig.savefig('exp/firstdiff_tonesweep%d.png'%(k))

######################################################################
# differenced_unitstep
fig, axs = plt.subplots(3,1,sharex=True)
n_step = np.linspace(-15,15,31,endpoint=True)
x_step = np.concatenate((np.zeros(15), np.ones(16)))
n_diff = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
h_diff = np.zeros(11)
h_diff[5]=1
h_diff[6]=-1
axs[0].stem(n_diff, h_diff, markerfmt='D',use_line_collection=True)
axs[0].set_title('Backward-Difference filter')
axs[1].stem(n_step, x_step, markerfmt='D',use_line_collection=True)
axs[1].set_title('Unit step')
y_step = np.convolve(x_step, h_diff, mode='same')
axs[2].stem(n_step, y_step, markerfmt='D',use_line_collection=True)
axs[2].set_title('Differenced Unit step')
fig.savefig('exp/differenced_unitstep.png')


