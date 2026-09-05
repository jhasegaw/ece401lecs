import numpy as np
import matplotlib.figure

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
fig.savefig('exp/triplebeat.png')


