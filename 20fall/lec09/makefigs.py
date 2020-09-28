import numpy  as np
import matplotlib.figure, subprocess

# periodic and aperiodic signals?
fig = matplotlib.figure.Figure(figsize=(7,4))
ax = fig.subplots()
nset = np.arange(1002)
xset = np.cos(0.04*np.pi*nset)*np.exp(-0.01*nset)
for k in range(0,10):
    nperiods = 10-k
    T0 = 2000/nperiods
    ax.clear()
    ax.plot([-1001,1001],[0,0],'k-',[0,1e-6],[-0.1,1.25],'k-')
    for m in range(-nperiods,nperiods):
        if m%2==0:
            ax.plot(int(m*T0/2)+nset[0:int(T0/2)],xset[0:int(T0/2)],'m-')
        else:
            ax.plot(int(m*T0/2)+nset[0:int(T0/2)],xset[int(T0/2):0:-1],'m-')
    ax.set_xlabel('n')
    ax.set_title('$x[n]$, periodic with period $T_0=%d$ samples'%(T0))
    fig.savefig('exp/aperiodic%d.png'%(k))
for k in range(10,13):
    fig.savefig('exp/aperiodic%d.png'%(k))
subprocess.call('convert -delay 50 -dispose previous exp/aperiodic?.png exp/aperiodic??.png exp/aperiodic.gif'.split())

# tall thin rectangle approximations of sum X[k] -> int X(w)?
fig = matplotlib.figure.Figure(figsize=(7,4))
ax = fig.subplots()
omega = np.linspace(0,2*np.pi,1000)
X = 1+np.sin(omega/2)
xbox=np.array([-0.5,-0.5+1e-6,0.5-1e-6,0.5])
ybox=np.array([0,1,1,0])
for logN in range(1,11):
    N = np.power(2,logN)
    kset = np.arange(N)
    Xk = 1+np.sin(np.pi*kset/N)
    ax.clear()
    ax.plot(omega,X,'k--')
    ax.plot([-np.pi/2,2*np.pi],[0,0],'k-',[0,1e-6],[-0.1,1.1],'k-')
    for k in kset:
        ax.plot((xbox+k)*2*np.pi/N,ybox*Xk[k])
    ax.set_xlabel('$ω$')
    ax.set_ylabel('$X(ω)$')
    ax.set_xticks(np.pi*np.array([-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2]))
    ax.set_xticklabels(['-π/4','0','π/4','π/2','3π/4','π','5π/4','3π/2','7π/8','2π'])
    ax.set_title('$\int_0^{2π}X(ω)dω ≈ \sum_{k=0}^{%d} (2π/%d) X[k]$'%(N-1,N))
    fig.tight_layout()
    fig.savefig('exp/integral%d.png'%(logN))
    
subprocess.call('convert -delay 50 -dispose previous exp/integral?.png exp/integral.gif'.split())

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
axs[0].set_title('$f[m]=δ[m]$')
axs[1].stem(gn,g,use_line_collection=True,markerfmt='gD')
axs[1].set_title('$g[m]=δ[m-3]$')
axs[2].set_title('$f[m]*g[m]$')
axs[2].set_xlabel('$m$')
fig.tight_layout()
for k in range(6,11):
    fig.savefig('exp/dconv%d.png'%(k))
for k in range(11,29):
    n = k-11-7
    axs[1].clear()
    axs[1].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
    axs[1].stem(n-gn,g,use_line_collection=True,markerfmt='gD')
    axs[1].set_title('$g[%d-m]$'%(n))
    axs[1].text(n,0.1,'n',FontSize=24)
    axs[2].clear()
    axs[2].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
    axs[2].stem(nset[:(k-7)],h[:(k-7)],use_line_collection=True,markerfmt='gD')
    axs[2].text(n,0.1,'n',FontSize=24)
    axs[2].set_title('$f[m]*g[m]$')
    axs[2].set_xlabel('$m$')
    fig.savefig('exp/dconv%d.png'%(k))
for k in range(29,35):
    fig.savefig('exp/dconv%d.png'%(k))    
subprocess.call('convert -delay 33 -dispose previous exp/dconv?.png exp/dconv??.png exp/dconv.gif'.split())
    
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
axs[0].set_title('$f[m]=δ[m-3]$')
axs[1].stem(gn,g,use_line_collection=True,markerfmt='gD')
axs[1].set_title('$g[m]=δ[m-3]$')
axs[2].set_title('$f[m]*g[m]$')
axs[2].set_xlabel('$m$')
fig.tight_layout()
for k in range(6,11):
    fig.savefig('exp/sdconv%d.png'%(k))
for k in range(11,29):
    n = k-11-7
    axs[1].clear()
    axs[1].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
    axs[1].stem(n-gn,g,use_line_collection=True,markerfmt='gD')
    axs[1].set_title('$g[%d-m]$'%(n))
    axs[1].text(n,0.1,'n',FontSize=24)
    axs[2].clear()
    axs[2].plot([-10.5,10.5],[0,0],'k-',[0,1e-6],[-0.1,2],'k-')
    axs[2].stem(nset[:(k-7)],h[:(k-7)],use_line_collection=True,markerfmt='gD')
    axs[2].text(n,0.1,'n',FontSize=24)
    axs[2].set_title('$f[m]*g[m]$')
    axs[2].set_xlabel('$m$')
    fig.savefig('exp/sdconv%d.png'%(k))
for k in range(29,35):
    fig.savefig('exp/sdconv%d.png'%(k))    
subprocess.call('convert -delay 33 -dispose previous exp/sdconv?.png exp/sdconv??.png exp/sdconv.gif'.split())
    
