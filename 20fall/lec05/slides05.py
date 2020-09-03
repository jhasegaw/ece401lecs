import matplotlib.figure
import numpy as np

fig = matplotlib.figure.Figure(figsize=(6, 6))
axs = fig.subplots(3,3,sharex=True,sharey=True)
t=np.linspace(0,1,201,endpoint=True)
x = [ t, np.cos(2*np.pi*t), np.sin(2*np.pi*t)  ]
title=['', 'cos(2πt)','sin(2πt)']

for m in range(1,3):
    axs[m,0].plot(t,x[m])
    axs[m,0].set_title(title[m])
    axs[0,m].plot(t,x[m])
    axs[0,m].set_title(title[m])
    for n in range(1,3):
        axs[m,n].plot(t,x[m]*x[n],[0,1],[0,0],'r--')
        axs[m,n].set_title('%s*%s'%(title[m],title[n]))
fig.savefig('exp/orthogonality_cos_sin.png')

fig = matplotlib.figure.Figure(figsize=(6, 6))
axs = fig.subplots(3,3,sharex=True,sharey=True)
t=np.linspace(0,1,201,endpoint=True)
x = [ t, np.sin(2*np.pi*3*t), np.sin(2*np.pi*4*t)  ]
title=['', 'sin(2π3t)','sin(2π4t)']

for m in range(1,3):
    axs[m,0].plot(t,x[m])
    axs[m,0].set_title(title[m])
    axs[0,m].plot(t,x[m])
    axs[0,m].set_title(title[m])
    for n in range(1,3):
        axs[m,n].plot(t,x[m]*x[n],[0,1],[0,0],'r--')
        axs[m,n].set_title('%s*%s'%(title[m],title[n]))
fig.savefig('exp/orthogonality_3_4.png')


fig = matplotlib.figure.Figure(figsize=(6, 6))
axs = fig.subplots(3,2,sharex=True,sharey=True)
t=np.linspace(0,1,201,endpoint=True)
y = [ t, np.sin(2*np.pi*3*t), np.sin(2*np.pi*4*t)  ]
ty=['', 'sin(2π3t)','sin(2π4t)']
x = 1.5*np.sin(2*np.pi*3*t)+0.25*np.sin(2*np.pi*4*t)
integral = ['', '1.5/2', '0.25/2']
axs[0,1].plot(t,x)
axs[0,1].set_title('x(t)=1.5sin(2π3t)+0.25sin(2π4t)')
for m in range(1,3):
    axs[m,0].plot(t,y[m])
    axs[m,0].set_title(ty[m])
    axs[m,1].plot(t,y[m]*x,[0,1],[0,0],'r--')
    axs[m,1].text(0.05,-0.3-0.7*m,'integral = %s'%(integral[m]))
    axs[m,1].set_title('x(t)%s'%(ty[m]))
fig.savefig('exp/orthogonality_for_spectrum.png')


##############################################3
# Fourier series
t=np.linspace(-1.5,1.5,3001,endpoint=True)
x = np.zeros(t.shape)
x[(-1.25<t)&(t<-0.75)]=1
x[(-0.25<t)&(t<0.25)]=1
x[(0.75<t)&(t<1.25)]=1

xax = np.zeros(t.shape)
yax = [np.array([-1e-6,1e-6]), np.array([-1.05,1.65])]
accum = np.zeros(len(t))
lasttitle = []
for k in range(0,7):
    fig = matplotlib.figure.Figure(figsize=(6, 6))
    fig.tight_layout()    
    axs = fig.subplots(4,1,sharex=True)
    axs[0].plot(t,x,'b-',t,xax,'k--',yax[0],yax[1],'k--')
    axs[0].set_title('Square Wave x(t)')
    c = np.cos(2*np.pi*k*t)
    axs[1].plot(t,c,'b-',t,xax,'k--',yax[0],yax[1],'k--')
    axs[1].set_title('cos(2π%dt)'%(k))
    axs[2].plot(t,c*x,'b-',t,xax,'k--',yax[0],yax[1],'k--')
    axs[2].set_title('cos(2π%dt)x(t)'%(k))
    integral = np.inner(c,x)/3001
    axs[2].text(-0.5,1.25,'integral = %2.2g'%(integral))
    accum += integral*c
    lasttitle.append('%2.2gcos(2π%dt)'%(integral,k))
    axs[3].plot(t,accum,'b-',t,xax,'k--',yax[0],yax[1],'k--')
    axs[3].set_title('+'.join(lasttitle))
    fig.savefig('exp/fourierseries%d.png'%(k))

