import numpy  as np
import matplotlib.figure, subprocess

#Video showing  convolution with an impulse?
nset = np.arange(21)-10
f = np.zeros(21)
f[10] = 1
gn = np.array([0,1,2,3])
g = np.array([0,0,0,1])
h = np.zeros(21)
h[13] = 1
fig = matplotlib.figure.Figure(figsize=(7,6))
axs = fig.subplots(3,1,sharex=True)
for axnum in range(3):
    axs[axnum].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
axs[0].stem(nset,f,use_line_collection=True,markerfmt='gD')
axs[0].set_title('$δ[m]$')
axs[1].stem(gn,g,use_line_collection=True,markerfmt='gD')
axs[1].set_title('$g[m]=δ[m-3]$')
axs[2].stem(nset,h,use_line_collection=True,markerfmt='gD')
axs[2].set_title('$δ[m]*g[m]$')
axs[2].set_xlabel('$m$')
fig.tight_layout()
fig.savefig('exp/dconv.png')    
    
#Video showing  convolution with a delayed impulse
nset = np.arange(21)-10
f = np.zeros(21)
f[13] = 1
gn = np.array([0,1,2,3])
g = np.array([0,0,0,1])
h = np.zeros(21)
h[16] = 1
fig = matplotlib.figure.Figure(figsize=(7,6))
axs = fig.subplots(3,1,sharex=True)
for axnum in range(3):
    axs[axnum].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
axs[0].stem(nset,f,use_line_collection=True,markerfmt='gD')
axs[0].set_title('$g[m]=δ[m-3]$')
axs[1].stem(gn,g,use_line_collection=True,markerfmt='gD')
axs[1].set_title('$g[m]=δ[m-3]$')
axs[2].stem(nset,h,use_line_collection=True,markerfmt='gD')
axs[2].set_title('$g[m]*g[m]$')
axs[2].set_xlabel('$m$')
fig.tight_layout()
fig.savefig('exp/sdconv.png')
    
