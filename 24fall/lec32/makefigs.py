import numpy as np
import matplotlib.figure, subprocess, os, wave, cmath

os.makedirs('exp',exist_ok=True)


################################################################################
# Probably globally useful stuff
zeromarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='none')
polemarker = matplotlib.markers.MarkerStyle(marker='x',fillstyle='none')
fillmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='full')
omega = np.linspace(0,np.pi,5000)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
ucx = np.cos(2*omega)
ucy = np.sin(2*omega)
def plot_circle(ax, pole=None, ptext='a'):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    if pole != None:
        for k in range(len(pole)):
            s = np.sign(np.imag(pole[k]))
            ax.scatter(x=np.real(pole[k]),y=np.imag(pole[k]),s=40,c='b',marker=polemarker)
            ax.text(x=np.real(pole[k])-0.1,y=np.imag(pole[k])-0.2*s-0.05,s=ptext[k])
    ax.set_aspect('equal')

def plot_spec(ax, omega, H):
    ax.plot(omega,np.zeros(len(omega)),'k-')
    ax.plot(omega,np.abs(H))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Frequency ($\omega$)')
    ax.set_title('$|H(\omega)|$')

def plot_convolution(axs, x, h, y, nset, n, yylim):
    '''
    h should be defined with the same resolution as nset, but not the same range:
    its range should be from 0 to max(nset)-min(nset).
    y should start out all zeros, but should be accumulated over time (as output of this func).
    Actually, if nset not integers, stem will give weird response.
    '''
    axs[0].stem(nset,x,use_line_collection=True)
    ylim = [ min(-0.1,1.1*np.amin(x)), max(1.1,1.1*np.amax(x))  ]
    axs[0].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[0].set_title('$x[m]=\delta[m]$')
    axs[1].clear()
    hplot = np.zeros(nset.shape)
    hplot[nset <= n] = h[n-nset[nset<=n]]
    axs[1].stem(nset,hplot,use_line_collection=True)
    ylim = [ min(-1.1,1.1*np.amin(h)), max(1.1,1.1*np.amax(h))  ]
    axs[1].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],ylim,'k-')
    axs[1].set_title('$h[%d-m]$'%(n))
    axs[2].clear()
    y[nset==n] = np.sum(hplot[nset%1==0]*x[nset%1==0])
    axs[2].stem(nset,y,use_line_collection=True)
    axs[2].plot(nset,np.zeros(nset.shape),'k-',[0,1e-6],1.1*np.array(yylim),'k-')
    axs[2].set_title('$y[m]=h[m]*x[m]$')
    axs[2].set_xlabel('$m$')
    fig.tight_layout()
    return(y)
                     
###########################################################################
# Video showing impulse response as system varies from underdamped to overdamped
fig = matplotlib.figure.Figure((10,4))
ax = fig.subplots()
nset = np.arange(30)
def impulseresponse(n,negb,c):
    discrim = negb**2 - 4*c
    if discrim==0:
        return (n+1)*((negb/2)**n)
    elif discrim > 0:
        p1 = negb/2 + np.sqrt(discrim)/2
        p2 = negb/2 - np.sqrt(discrim)/2
        C2 = 1/(1-p1/p2)
        C1 = 1/(1-p2/p1)
        return C1*(p1**n) + C2*(p2**n)
    else:
        p = negb/2 + 1j*np.sqrt(-discrim)/2
        sigma = -np.log(np.abs(p))
        omega = np.angle(p)
        return np.exp(-sigma*n)*np.sin(omega*(n+1))/np.sin(omega)

cset = np.linspace(0.01,0.99,99)
for n in range(len(cset)):
    c = cset[n]
    h = impulseresponse(nset, 1, c)
    ax.clear()
    ax.stem(nset, h)
    ax.set_title('Impulse Response of $y[n]=x[n]+ y[n-1]- %2.2g y[n-2]$'%(c))
    fig.savefig('exp/damping-%d.png'%(n))
