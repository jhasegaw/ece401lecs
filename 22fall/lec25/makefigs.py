import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

os.makedirs('exp',exist_ok=True)

def draw_a_box(ax):
    '''Draw a box with arrows going in and out'''
    ax.set_xlim(0,1)
    ax.set_ylim(-0.5,0.5)
    ax.plot([0,0.4,0.38,0.4,0.38,0.4],[0,0,0.02,0,-0.02,0],'k')
    ax.plot(0.6+np.array([0,0.4,0.38,0.4,0.38,0.4]),[0,0,0.02,0,-0.02,0],'k')
    ax.plot([0.4,0.4,0.6,0.6,0.4],[-0.2,0.2,0.2,-0.2,-0.2],'k')

def make_invisible(a):
    a.tick_params(bottom=False, top=False, left=False, right=False)
    a.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    
    
##########################################################################################3
# impulseresponses[0:127].png
#  Let's look more closely at what convolution is.
#  Each sample of x[n] generates an impulse response.
#  Those impulse responses are added together to make the output.

fig = plt.figure(figsize=(6,4))
axs = fig.subplots(3,3)
L = 33
M = 32
N = L+M-1
x = np.random.randn(M)
xp = np.concatenate((x, np.zeros(N-M)))
h = np.exp(-0.05*np.arange(L))
y = np.convolve(x,h)
ymax = np.amax(np.abs(y))*1.1
xmax = np.amax(np.abs(x))*1.1

# Draw x[n] in axs[0,0], draw a box in axs[1,1]
draw_a_box(axs[1,1])
for a in [ axs[2,0], axs[0,1], axs[1,1], axs[2,1], axs[0,2] ]:
    make_invisible(a)

# Draw each delta of x, then its corresponding h, then update yp
yt = np.zeros(N)
for n in range(M):
    # empty impulse and output plots
    axs[0,0].clear()
    axs[0,0].plot(np.arange(N),xp,np.arange(N),np.zeros(N),'k--')
    axs[0,0].set_ylim(-xmax, xmax)
    axs[0,0].set_xlim(0,N)
    axs[0,0].set_title('$x[n]$')
    axs[1,0].clear()
    axs[1,0].plot(np.arange(N),np.zeros(N),'k--')
    axs[1,0].set_ylim(-xmax,xmax)
    axs[1,0].set_xlim(0,N)
    axs[1,0].set_title('$x[%d]\delta[n-%d]$'%(n,n))
    axs[1,2].clear()
    axs[1,2].plot(np.arange(N),np.zeros(N),'k--')
    axs[1,2].set_ylim(-ymax, ymax)
    axs[1,2].set_xlim(0,N)
    axs[1,2].set_title('$x[%d]h[n-%d]$'%(n,n))
    axs[2,2].clear()
    axs[2,2].plot(np.arange(N),yt,np.arange(N),np.zeros(N),'k--')
    axs[2,2].set_ylim(-ymax, ymax)
    axs[2,2].set_xlim(0,N)
    axs[2,2].set_title('$y[n]=Σx[m]h[n-m]$')
    fig.tight_layout()
    fig.savefig('exp/impulseresponses%d.png'%(4*n))

    # Add x[n] times delta
    markerline, stemlines, baselines = axs[0,0].stem(n, x[n], markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))    
    axs[0,0].set_ylim(-xmax, xmax)
    axs[0,0].set_xlim(0,N)
    markerline, stemlines, baselines = axs[1,0].stem(n, x[n], markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))    
    axs[1,0].set_ylim(-xmax, xmax)
    axs[1,0].set_xlim(0,N)
    fig.tight_layout()
    fig.savefig('exp/impulseresponses%d.png'%(4*n+1))

    # Add x[n] times h[m-n]
    markerline, stemlines, baselines = axs[1,2].stem(np.arange(n,n+L), x[n]*h, markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))    
    axs[1,2].set_ylim(-ymax, ymax)
    axs[1,2].set_xlim(0,N)
    yt[n:n+L] += x[n]*h
    markerline, stemlines, baselines = axs[2,2].stem(np.arange(n,n+L), yt[n:n+L], markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))    
    axs[2,2].set_ylim(-ymax, ymax)
    axs[2,2].set_xlim(0,N)
    fig.tight_layout()
    fig.savefig('exp/impulseresponses%d.png'%(4*n+2))
    
    # Add this to the plot of yt
    axs[2,2].clear()
    axs[2,2].plot(np.arange(N),yt,np.arange(N),np.zeros(N),'k--')
    axs[2,2].set_ylim(-ymax, ymax)
    axs[2,2].set_xlim(0,N)
    axs[2,2].set_title('$y[n]=Σx[m]h[n-m]$')
    fig.tight_layout()
    fig.savefig('exp/impulseresponses%d.png'%(4*n+3))
    

##########################################################################################3
# overlapadd[0:400].png
#  Here are all of the impulse responses generated by the samples in one frame of audio.
#  And here are all of the impulse responses generated by the samples in another frame.
#  When we add them together, we get the whole output.

fig = plt.figure(figsize=(6,4))
axs = fig.subplots(3,3)
L = 33
M = 32
N = L+2*M-1
x = np.random.randn(N)
h = np.exp(-0.05*np.arange(L))
y = np.convolve(x,h)
ymax = np.amax(np.abs(y))*1.1
xmax = np.amax(np.abs(x))*1.1

# Draw boxes in all center column
for a in [ axs[0,1], axs[1,1], axs[2,1] ]:
    make_invisible(a)
    draw_a_box(a)

# Draw each delta of x in its own frame and in overall x, then impulse responses
x1 = np.zeros(N)
x2 = np.zeros(N)
x3 = np.zeros(N)
y1 = np.zeros(N)
y2 = np.zeros(N)
y3 = np.zeros(N)
for m in [0,1,2]:
    axs[m,0].clear()
    axs[m,0].plot(np.arange(N),np.zeros(N),'k--')
    axs[m,0].set_ylim(-xmax, xmax)
    axs[m,0].set_xlim(0,N)
    axs[m,2].clear()
    axs[m,2].plot(np.arange(N),np.zeros(N),'k--')
    axs[m,2].set_ylim(-ymax, ymax)
    axs[m,2].set_xlim(0,N)
axs[0,0].set_title('First input frame, $x_1[n]$')
axs[1,0].set_title('Second input frame, $x_2[n]$')
axs[2,0].set_title('$x[n]$ total, all frames')
axs[0,2].set_title('First output frame, $y_1[n]$')
axs[1,2].set_title('Second output frame, $y_2[n]$')
axs[2,2].set_title('$y[n]$ total, all frames')
        
for n in range(2*M):
    # Add x[n] times delta
    axs[2,0].clear()
    axs[2,0].plot(x3)
    markerline, stemlines, baseline = axs[2,0].stem(n, x[n], markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
    axs[2,0].set_ylim(-xmax, xmax)
    axs[2,0].set_xlim(0,N)
    axs[2,0].set_title('$x[n]$ total, all frames')
    axs[2,0].set_xlabel('$n$')
    if n < M:
        axs[0,0].clear()
        axs[0,0].plot(x1)
        markerline, stemlines, baseline = axs[0,0].stem(n, x[n], markerfmt='ro')
        plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
        axs[0,0].set_ylim(-xmax,xmax)
        axs[0,0].set_xlim(0,N)
        axs[0,0].set_title('First input frame, $x_1[n]$')
    else:
        axs[1,0].clear()
        axs[1,0].plot(x2)
        markerline, stemlines, baseline = axs[1,0].stem(n, x[n], markerfmt='ro')
        plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
        axs[1,0].set_ylim(-xmax,xmax)
        axs[1,0].set_xlim(0,N)
        axs[1,0].set_title('Second input frame, $x_2[n]$')
    fig.tight_layout()
    fig.savefig('exp/overlapadd%d.png'%(4*n))

    # Add x[n] times h[m-n], with new stuff in red
    axs[2,2].clear()
    axs[2,2].plot(y3, 'k')
    markerline, stemlines, baseline = axs[2,2].stem(np.arange(n,n+L), x[n]*h, markerfmt='ro')
    plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
    axs[2,2].set_ylim(-ymax,ymax)
    axs[2,2].set_xlim(0,N)
    axs[2,2].set_title('$y[n]$ total, all frames')
    axs[2,2].set_xlabel('$n$')
    if n < M:
        axs[0,2].clear()
        axs[0,2].plot(y1, 'ko')
        markerline, stemlines, baseline = axs[0,2].stem(np.arange(n,n+L), x[n]*h, markerfmt='ro')
        plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
        axs[0,2].set_ylim(-ymax,ymax)
        axs[0,2].set_xlim(0,N)
        axs[0,2].set_title('First output frame, $y_1[n]$')
    else:
        axs[1,2].clear()
        axs[1,2].plot(y2, 'k')
        markerline, stemlines, baseline = axs[1,2].stem(np.arange(n,n+L), x[n]*h, markerfmt='ro')
        plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
        axs[1,2].set_ylim(-ymax,ymax)
        axs[1,2].set_xlim(0,N)
        axs[1,2].set_title('Second output frame, $y_2[n]$')
    fig.tight_layout()
    fig.savefig('exp/overlapadd%d.png'%(4*n+1))

    # Add x[n] times delta
    x3[n] = x[n]
    axs[2,0].clear()
    axs[2,0].plot(x3)
    axs[2,0].set_title('$x[n]$ total, all frames')
    axs[2,0].set_ylim(-xmax,xmax)
    axs[2,0].set_xlim(0,N)
    axs[2,0].set_xlabel('$n$')
    if n < M:
        axs[0,0].clear()
        x1[n] = x[n]
        axs[0,0].plot(x1)
        axs[0,0].set_ylim(-xmax,xmax)
        axs[0,0].set_xlim(0,N)
        axs[0,0].set_title('First input frame, $x_1[n]$')
    else:
        axs[1,0].clear()
        x2[n] = x[n]
        axs[1,0].plot(x2)
        axs[1,0].set_ylim(-xmax,xmax)
        axs[1,0].set_xlim(0,N)
        axs[1,0].set_title('Second input frame, $x_2[n]$')
    fig.tight_layout()
    fig.savefig('exp/overlapadd%d.png'%(4*n+2))

    # Add x[n] times h[m-n], with new stuff in red
    y3[n:n+L] += x[n]*h
    axs[2,2].clear()
    axs[2,2].plot(y3, 'k')
    axs[2,2].set_title('$y[n]$ total, all frames')
    axs[2,2].set_ylim(-ymax,ymax)
    axs[2,2].set_xlim(0,N)
    axs[2,2].set_xlabel('$n$')
    if n < M:
        axs[0,2].clear()
        y1[n:n+L] += x[n]*h
        axs[0,2].plot(y1, 'k')
        axs[0,2].set_ylim(-ymax,ymax)
        axs[0,2].set_xlim(0,N)
        axs[0,2].set_title('First output frame, $y_1[n]$')
    else:
        y2[n:n+L] += x[n]*h
        axs[1,2].clear()
        axs[1,2].plot(y2, 'k')
        axs[1,2].set_ylim(-ymax,ymax)
        axs[1,2].set_xlim(0,N)
        axs[1,2].set_title('Second output frame, $y_2[n]$')
    fig.tight_layout()
    fig.savefig('exp/overlapadd%d.png'%(4*n+3))
