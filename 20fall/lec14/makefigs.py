import numpy  as np
import matplotlib.figure, subprocess, os

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
def plot_circle(ax, zero=None, ztext='-b', pole=None, ptext='a'):
    ax.plot([0,1e-6],[-2,2],'k-',[-2,2],[0,0],'k-')
    ax.text(1.5,0,'Real(z)')
    ax.text(0,1.9,'Imag(z)')
    ax.plot(ucx,ucy,'k-')
    if zero != None:
        for k in range(len(zero)):
            s = np.sign(np.imag(zero[k]))
            ax.scatter(x=np.real(zero[k]),y=np.imag(zero[k]),s=40,c='r',marker=zeromarker)
            ax.text(x=np.real(zero[k])+0.1,y=np.imag(zero[k])+0.15*s-0.05,s=ztext[k])
    if pole != None:
        for k in range(len(pole)):
            s = np.sign(np.imag(pole[k]))
            ax.scatter(x=np.real(pole[k]),y=np.imag(pole[k]),s=40,c='b',marker=polemarker)
            ax.text(x=np.real(pole[k])-0.1,y=np.imag(pole[k])-0.2*s-0.05,s=ptext[k])

def plot_spec(ax, omega, H):
    ax.plot(omega,np.zeros(len(omega)),'k-')
    ax.plot(omega,np.abs(H))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Frequency ($\omega$)')
    ax.set_title('$|H(\omega)|$')
    
            
################################################################################
# onezeroresponse
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
nb = np.exp(2j*np.pi/5)
H = np.abs(1-nb*np.exp(-1j*omega))
for m in range(50):
    n=100*m
    axs[0].clear()
    plot_circle(axs[0], zero=[nb], ztext=['-b'])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    axs[0].plot([np.real(nb),np.cos(omega[n])],[np.imag(nb),np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'r-')
    fig.savefig('exp/onezeroresponse%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/onezeroresponse?.png exp/onezeroresponse??.png exp/onezeroresponse.gif'.split())


################################################################################
# twozeroresponse
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
nb = [ np.exp(2j*np.pi/5), np.exp(-2j*np.pi/5) ]
H = np.abs(1-nb[0]*np.exp(-1j*omega))*np.abs(1-nb[1]*np.exp(-1j*omega))
for m in range(50):
    n=100*m
    axs[0].clear()
    plot_circle(axs[0], zero=nb, ztext=['$r_1$','$r_2$'])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    for k in range(len(nb)):
        axs[0].plot([np.real(nb[k]),np.cos(omega[n])],[np.imag(nb[k]),np.sin(omega[n])],'r-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'r-')
    fig.savefig('exp/twozeroresponse%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/twozeroresponse?.png exp/twozeroresponse??.png exp/twozeroresponse.gif'.split())

################################################################################
# onezeronotch
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
nb = np.exp(2j*np.pi/5)
a = np.exp(-0.1)*np.exp(2j*np.pi/5)
H = np.abs(1-nb*np.exp(-1j*omega))/np.abs(1-a*np.exp(-1j*omega))
for m in range(50):
    n=100*m
    axs[0].clear()
    plot_circle(axs[0], zero=[nb], ztext=['$r$'],pole=[a], ptext=['$p$'])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    axs[0].plot([np.real(nb),np.cos(omega[n])],[np.imag(nb),np.sin(omega[n])],'r-')
    axs[0].plot([np.real(a),np.cos(omega[n])],[np.imag(a),np.sin(omega[n])],'b-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'m-')
    fig.savefig('exp/onezeronotch%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/onezeronotch?.png exp/onezeronotch??.png exp/onezeronotch.gif'.split())

################################################################################
#   Image showing pole-zero plot and magnitude response of a filter with BW=1, a=exp(-0.5)
for bw in [1.0, 0.2]:
    fig = matplotlib.figure.Figure((10,4))
    axs = fig.subplots(1,2)
    nb = np.exp(2j*np.pi/5)
    a = np.exp(-bw/2)*np.exp(2j*np.pi/5)
    H = np.abs(1-nb*np.exp(-1j*omega))/np.abs(1-a*np.exp(-1j*omega))
    axs[0].clear()
    plot_circle(axs[0], zero=[nb], ztext=['$r$'],pole=[a], ptext=['$p$'])
    axs[0].set_aspect('equal')
    axs[0].set_title('Pole-Zero Plot, $a=e^{-%2.1f}$'%(bw/2))
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].set_title('Magnitude Response, Bandwidth=%2.1f radian/sample'%(bw))
    fig.savefig('exp/notch%dbw.png'%(int(100*bw)))

################################################################################
# twozeronotch
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(1,2)
nb = [np.exp(2j*np.pi/5), np.exp(-2j*np.pi/5)]
a = [np.exp(-0.1)*np.exp(2j*np.pi/5), np.exp(-0.1)*np.exp(-2j*np.pi/5)]
H = np.ones(omega.shape)
for k in range(len(a)):
    H *= np.abs(1-nb[k]*np.exp(-1j*omega))/np.abs(1-a[k]*np.exp(-1j*omega))
for m in range(50):
    n=100*m
    axs[0].clear()
    plot_circle(axs[0], zero=nb, ztext=['$r_1$','$r_2$'],pole=a, ptext=['$p_1$','$p_2$'])
    axs[0].scatter(x=np.cos(omega[n]),y=np.sin(omega[n]),s=40,marker=fillmarker)
    for k in range(len(nb)):
        axs[0].plot([np.real(nb[k]),np.cos(omega[n])],[np.imag(nb[k]),np.sin(omega[n])],'r-')
        axs[0].plot([np.real(a[k]),np.cos(omega[n])],[np.imag(a[k]),np.sin(omega[n])],'b-')
    axs[0].set_aspect('equal')
    axs[1].clear()
    plot_spec(axs[1], omega, H)
    axs[1].scatter(x=omega[n],y=H[n],s=40,marker=fillmarker)
    axs[1].plot([omega[n]-1e-6,omega[n]],[0,H[n]],'m-')
    fig.savefig('exp/twozeronotch%d.png'%(m))

subprocess.call('convert -delay 10 -dispose previous exp/twozeronotch?.png exp/twozeronotch??.png exp/twozeronotch.gif'.split())

