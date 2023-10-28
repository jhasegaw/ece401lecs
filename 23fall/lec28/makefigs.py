import numpy  as np
import matplotlib.figure, subprocess


################################################################################
#  Picture showing $|H(\omega)|=\left|\frac{1+0.3z^{-1}}{1-0.8z^{-1}}\right|$
fig = matplotlib.figure.Figure((10,4))
ax = fig.subplots()
omega = np.linspace(0,np.pi,100)
xticks = np.pi*np.arange(0,5)/4
xticklabels=['0','π/4','π/2','3π/4','π']
nb = -0.3
a = 0.8
H = np.abs((1-nb*np.exp(1j*omega))/(1-a*np.exp(1j*omega)))
ax.plot(omega,H)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Frequency ($\omega$)')
ax.set_title('$|H(\omega)|=(1+0.3e^{j\omega})/(1-0.8e^{-j\omega})$')
fig.savefig('exp/intro.png')


################################################################################
#  picture showing  $a^n u[n]$ for positive, negative, and complex stable values of $a$
a = [0.9, -0.9, 0.9*np.exp(1j*np.pi/5)]
atext = ['0.9','-0.9','0.9e^{jπ/5}']
nset = np.arange(-5,30)
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1)
h = np.zeros(len(nset),dtype=np.complex64)
for k in range(3):
    h[nset >= 0] = np.power(a[k], nset[nset>=0])
    axs[k].plot(nset,np.zeros(len(nset)),'k-')
    axs[k].stem(nset,np.real(h),use_line_collection=True)
    axs[k].set_title('$h[n]=(%s)^nu[n]$'%(atext[k]))
    if k==2:
        axs[k].plot(nset,np.imag(h),'b--')
        axs[k].set_title('$h[n]=(%s)^nu[n]$ (Imaginary part dashed)'%(atext[k]))
fig.tight_layout()
fig.savefig('exp/iir_stable.png')

################################################################################
#  Picture showing impulse responses and frequency responses for positive, negative, and complex $a$
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,2)
for k in range(3):
    h[nset >= 0] = np.power(a[k], nset[nset>=0])
    axs[k,0].plot(nset,np.zeros(len(nset)),'k-')
    axs[k,0].stem(nset,np.real(h),use_line_collection=True)
    axs[k,0].set_title('$h[n]=(%s)^nu[n]$'%(atext[k]))
    if k==2:
        axs[k,0].plot(nset,np.imag(h),'b--')
        axs[k,0].set_title('$h[n]=(%s)^nu[n]$ (Imaginary part dashed)'%(atext[k]))
    H = np.abs(1/(1-a[k]*np.exp(-1j*omega)))
    axs[k,1].plot(omega,H)
    axs[k,1].set_title('$H(\omega)=1/(1-(%s)e^{-j\omega})$'%(atext[k]))
fig.tight_layout()
fig.savefig('exp/iir_freqresponse.png')

################################################################################
#  picture showing $a^n u[n]$ for positive, negative, and complex unstable values of $a$
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1)
a = [1.1, -1.1, 1.1*np.exp(1j*np.pi/5)]
atext = ['1.1','-1.1','1.1e^{jπ/5}']
for k in range(3):
    h[nset >= 0] = np.power(a[k], nset[nset>=0])
    axs[k].plot(nset,np.zeros(len(nset)),'k-')
    axs[k].stem(nset,np.real(h),'b-',use_line_collection=True)
    axs[k].set_title('$h[n]=(%s)^nu[n]$'%(atext[k]))
    if k==2:
        axs[k].plot(nset,np.imag(h),'b--')
        axs[k].set_title('$h[n]=(%s)^nu[n]$ (Imaginary part dashed)'%(atext[k]))
fig.tight_layout()
fig.savefig('exp/iir_unstable.png')

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
    fig.savefig('exp/magresponse%d.png'%(n))

subprocess.call('convert -delay 10 -dispose previous exp/magresponse?.png exp/magresponse??.png exp/magresponse.gif'.split())


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
    fig.savefig('exp/toneresponse%d.png'%(n))
    
subprocess.call('convert -delay 10 -dispose previous exp/toneresponse?.png exp/toneresponse??.png exp/toneresponse.gif'.split())
