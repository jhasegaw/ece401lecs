import numpy  as np
import matplotlib.figure, subprocess

################################################################################
#  Video showing signal as a weighted sum of impulses?
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)
N = 30
nset = np.arange(N)
x = np.convolve(np.random.randn(N),[1,0.3],mode='same')
ylim = [ np.amin(x)-0.1, np.amax(x)+0.1 ]
axs[0].stem(nset,x)
axs[0].set_title('$x[n]$')
for n in range(N):
    axs[1].clear()
    axs[1].plot(nset,np.zeros(len(nset)),'r-')
    axs[1].stem([n],[x[n]])
    axs[1].set_title('%2.2f $\delta[n-%d]$'%(x[n],n))
    axs[1].set_ylim(ylim)
    fig.tight_layout()
    fig.savefig('exp/impulses%d.png'%(2*n))
    axs[2].clear()
    axs[2].stem(nset[:(n+1)],x[:(n+1)])
    axs[2].set_title('$\sum_{m=0}^{%d} x[m]\delta[n-m]$'%(n))
    axs[2].set_ylim(ylim)
    fig.tight_layout()
    fig.savefig('exp/impulses%d.png'%(2*n+1))

subprocess.call('convert -delay 20 -dispose previous exp/impulses?.png exp/impulses??.png exp/impulses.gif'.split())


################################################################################
# Video showing z traveling around the unit circle, and |H(w)| with a dip at the
# frequency closest to the zero.
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
omega = np.linspace(0,np.pi,100)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
H = np.abs(1 - 2*np.exp(1j*omega)+2*np.exp(2*1j*omega))
Phi = np.angle(1 - 2*np.exp(1j*omega)+2*np.exp(2*1j*omega))/np.pi
ucx = np.cos(2*omega)
ucy = np.sin(2*omega)
z1 = np.array([[1, 1],[1, -1]])
openmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='none')
fillmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='full')
def plot_circle(ax):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    ax.scatter(x=z1[0],y=z1[1],s=20,marker=openmarker)
    ax.text(x=z1[0,0],y=z1[1,0],s='$z_1$')
    ax.text(x=z1[0,1],y=z1[1,1],s='$z_2$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=20,marker=fillmarker)
    axs[0].plot([z1[0,0],np.cos(omega[n])],[z1[1,0],np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    axs[1].plot(omega,np.zeros(len(omega)),'k-')
    axs[1].plot(omega,H)
    axs[1].scatter(x=omega[n],y=H[n],s=20,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'r-')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_xlabel('Frequency ($\omega$)')
    axs[1].set_title('$|H(\omega)|$')
    fig.savefig('exp/magresponse%d.png'%(n))

subprocess.call('convert -delay 10 -dispose previous exp/magresponse?.png exp/magresponse??.png exp/magresponse.gif'.split())


################################################################################
# Video showing z traveling around the unit circle as $x[n]$ changes frequency,
# showing that $y[n]$ has its lowest amplitude when $\omega=0.61\pi$.
fig = matplotlib.figure.Figure((14,4))
gs = fig.add_gridspec(3,9)
axs = [ fig.add_subplot(gs[:,0:3]), fig.add_subplot(gs[0,3:]),
        fig.add_subplot(gs[1,3:]), fig.add_subplot(gs[2,3:]) ]
nset = np.linspace(0,20,201)
extran = np.linspace(-2,20,221)
h = [1,-2,2]
hlong = np.array([1,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,2])
axs[2].stem([0,1,2],h)
axs[2].plot(nset,np.zeros(len(nset)),'k-')
axs[2].set_title('$h[n]$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=20,marker=fillmarker)
    axs[0].plot([z1[0,0],np.cos(omega[n])],[z1[1,0],np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    x = np.cos(nset*omega[n])
    axs[1].plot(nset,np.zeros(len(nset)),'k-',nset,np.cos(nset*omega[n]),'b-')
    axs[1].set_title('$x[n]=cos(%2.2fπn)$'%(n/100))
    axs[1].set_ylim([-1.1,1.1])
    axs[3].clear()
    y = np.convolve(np.cos(extran*omega[n]),hlong,mode='valid')
    axs[3].plot(nset,np.zeros(len(nset)),'k-',nset,y,'b-')
    axs[3].set_ylim([-5.1,5.1])
    axs[3].set_title('$y[n]=h[n]*x[n] = %2.2f cos(%2.2fπn + %2.2fπ)$'%(H[n],n/100,Phi[n]))
    fig.tight_layout()
    fig.savefig('exp/toneresponse%d.png'%(n))
    
subprocess.call('convert -delay 10 -dispose previous exp/toneresponse?.png exp/toneresponse??.png exp/toneresponse.gif'.split())
################################################################################
# Video showing z traveling around the unit circle, and |H(w)| with a dip at the
# frequency closest to the zero.
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
omega = np.linspace(0,np.pi,100)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
H = np.abs(1 - 2*np.exp(1j*omega)+2*np.exp(2*1j*omega))
Phi = np.angle(1 - 2*np.exp(1j*omega)+2*np.exp(2*1j*omega))/np.pi
ucx = np.cos(2*omega)
ucy = np.sin(2*omega)
z1 = np.array([[1, 1],[1, -1]])
openmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='none')
fillmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='full')
def plot_circle(ax):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    ax.scatter(x=z1[0],y=z1[1],s=20,marker=openmarker)
    ax.text(x=z1[0,0],y=z1[1,0],s='$z_1$')
    ax.text(x=z1[0,1],y=z1[1,1],s='$z_2$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=20,marker=fillmarker)
    axs[0].plot([z1[0,0],np.cos(omega[n])],[z1[1,0],np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    axs[1].plot(omega,np.zeros(len(omega)),'k-')
    axs[1].plot(omega,H)
    axs[1].scatter(x=omega[n],y=H[n],s=20,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'r-')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_xlabel('Frequency ($\omega$)')
    axs[1].set_title('$|H(\omega)|$')
    fig.savefig('exp/magresponse%d.png'%(n))

subprocess.call('convert -delay 10 -dispose previous exp/magresponse?.png exp/magresponse??.png exp/magresponse.gif'.split())


################################################################################
# Video showing z traveling around the unit circle as $x[n]$ changes frequency,
# showing that $y[n]$ has its lowest amplitude when $\omega=0.61\pi$.
fig = matplotlib.figure.Figure((14,4))
gs = fig.add_gridspec(3,9)
axs = [ fig.add_subplot(gs[:,0:3]), fig.add_subplot(gs[0,3:]),
        fig.add_subplot(gs[1,3:]), fig.add_subplot(gs[2,3:]) ]
nset = np.linspace(0,20,201)
extran = np.linspace(-2,20,221)
h = [1,-2,2]
hlong = np.array([1,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,2])
axs[2].stem([0,1,2],h)
axs[2].plot(nset,np.zeros(len(nset)),'k-')
axs[2].set_title('$h[n]$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=20,marker=fillmarker)
    axs[0].plot([z1[0,0],np.cos(omega[n])],[z1[1,0],np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    x = np.cos(nset*omega[n])
    axs[1].plot(nset,np.zeros(len(nset)),'k-',nset,np.cos(nset*omega[n]),'b-')
    axs[1].set_title('$x[n]=cos(%2.2fπn)$'%(n/100))
    axs[1].set_ylim([-1.1,1.1])
    axs[3].clear()
    y = np.convolve(np.cos(extran*omega[n]),hlong,mode='valid')
    axs[3].plot(nset,np.zeros(len(nset)),'k-',nset,y,'b-')
    axs[3].set_ylim([-5.1,5.1])
    axs[3].set_title('$y[n]=h[n]*x[n] = %2.2f cos(%2.2fπn + %2.2fπ)$'%(H[n],n/100,Phi[n]))
    fig.tight_layout()
    fig.savefig('exp/toneresponse%d.png'%(n))
    
subprocess.call('convert -delay 10 -dispose previous exp/toneresponse?.png exp/toneresponse??.png exp/toneresponse.gif'.split())

################################################################################
# Video showing z traveling around the unit circle, and |H(w)| with a dip at the
# frequency closest to the zero.
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
omega = np.linspace(0,np.pi,100)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
a = 0.9*np.exp(3j*np.pi/5)
nb = 0.9*np.exp(2j*np.pi/5)
H = np.abs((1-nb*np.exp(-1j*omega))/(1-a*np.exp(-1j*omega)))
Phi = np.angle((1-nb*np.exp(-1j*omega))/(1-a*np.exp(-1j*omega)))
ucx = np.cos(2*omega)
ucy = np.sin(2*omega)
zeromarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='none')
polemarker = matplotlib.markers.MarkerStyle(marker='x',fillstyle='none')
fillmarker = matplotlib.markers.MarkerStyle(marker='o',fillstyle='full')
def plot_circle(ax):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    ax.scatter(x=np.real(nb),y=np.imag(nb),s=40,c='r',marker=zeromarker)
    ax.scatter(x=np.real(a),y=np.imag(a),s=40,c='b',marker=polemarker)
    ax.text(x=np.real(nb)-0.05,y=np.imag(nb)+0.15,s='$-b$')
    ax.text(x=np.real(a)-0.05,y=np.imag(a)+0.15,s='$a$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    axs[0].plot([np.real(nb),np.cos(omega[n])],[np.imag(nb),np.sin(omega[n])],'r-')
    axs[0].plot([np.real(a),np.cos(omega[n])],[np.imag(a),np.sin(omega[n])],'b-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    axs[1].plot(omega,np.zeros(len(omega)),'k-')
    axs[1].plot(omega,H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'m-')
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_xlabel('Frequency ($\omega$)')
    axs[1].set_title('$|H(\omega)|$')
    fig.savefig('exp/mag2response%d.png'%(n))

subprocess.call('convert -delay 10 -dispose previous exp/mag2response?.png exp/mag2response??.png exp/mag2response.gif'.split())


################################################################################
# Video showing z traveling around the unit circle as $x[n]$ changes frequency,
# showing that $y[n]$ has its lowest amplitude when $\omega=0.61\pi$.
fig = matplotlib.figure.Figure((14,4))
gs = fig.add_gridspec(3,9)
axs = [ fig.add_subplot(gs[:,0:3]), fig.add_subplot(gs[0,3:]),
        fig.add_subplot(gs[1,3:]), fig.add_subplot(gs[2,3:]) ]
nset = np.linspace(-20,20,401)
extran = np.linspace(-22,20,421)
h = np.zeros(401)
h[nset==0] = 1
h[nset > 0] = np.real((a-nb)*np.power(a,nset[nset>0]))
axs[2].plot(nset,h)
axs[2].plot(nset,np.zeros(len(nset)),'k-')
axs[2].set_title('Real part of $h[n]$')
for n in range(len(omega)):
    axs[0].clear()
    plot_circle(axs[0])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    axs[0].plot([np.real(nb),np.cos(omega[n])],[np.imag(nb),np.sin(omega[n])],'r-')
    axs[0].plot([np.real(a),np.cos(omega[n])],[np.imag(a),np.sin(omega[n])],'b-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    x = np.cos(nset*omega[n])
    axs[1].plot(nset,np.zeros(len(nset)),'k-',nset,np.cos(nset*omega[n]),'b-')
    axs[1].set_title('$x[n]=cos(%2.2fπn)$'%(n/100))
    axs[1].set_ylim([-1.1,1.1])
    axs[3].clear()
    y = H[n]*np.cos(nset*omega[n]+Phi[n])
    axs[3].plot(nset,np.zeros(len(nset)),'k-',nset,y,'b-')
    axs[3].set_ylim([-5.1,5.1])
    axs[3].set_title('Real part of $y[n]=h[n]*x[n]$, which is $%2.2f cos(%2.2fπn + %2.2fπ)$'%(H[n],n/100,Phi[n]))
    fig.tight_layout()
    fig.savefig('exp/tone2response%d.png'%(n))
    
subprocess.call('convert -delay 10 -dispose previous exp/tone2response?.png exp/tone2response??.png exp/tone2response.gif'.split())

