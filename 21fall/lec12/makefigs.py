import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans

os.makedirs('exp',exist_ok=True)

###########################################################################
# convolutionproof.png
n = np.arange(-3,8)
x = np.zeros(len(n))
x[3:7] = np.array([0.7,0.3,-0.99,0.5])
h = (np.exp(-0.5*(np.arange(-3,4)*np.arange(-3,4)))-np.exp(-0.5*(np.arange(-4,3)*np.arange(-4,3))))
fig, axes = plt.subplots(6,2,figsize=(14,10))
for m in range(4):
    delta = np.zeros(len(n))
    delta[3+m] = x[3+m]
    hshift = np.convolve(delta,h,'same')
    axes[m,0].stem(n,delta)
    axes[m,0].set_ylim(-1,1)
    axes[m,0].set_title('$x[%d]\delta[n-%d]$'%(m,m))
    axes[m,1].stem(n,hshift)
    axes[m,1].set_ylim(-1,1)
    axes[m,1].set_title('$x[%d]h[n-%d]$'%(m,m))
ax0 = plt.subplot(325)
ax0.stem(n,x)
ax0.set_title('$x[n]=\sum_m x[m]\delta[n-m]$')
ax0.set_ylim(-1,1)
ax1 = plt.subplot(326)
ax1.stem(n,np.convolve(x,h,'same'))
ax1.set_title('$y[n]=\sum_m x[m]h[n-m]$')
ax1.set_ylim(-1,1)

fig.tight_layout()

# Get the bounding boxes of the axes including text decorations
#r = fig.canvas.get_renderer()
#get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
#bboxes = np.array(list(map(get_bbox, axes.flat)),mtrans.Bbox).reshape(axes.shape)

#Get the minimum and maximum extent, get the coordinate half-way between those
#ymax = np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
#ymin = np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
#ys = np.c_[ymax[1:], ymin[:-1]].mean(axis=1)

# Draw a horizontal lines at those coordinates
#y=ys[-2]
#line = plt.Line2D([0,1],[y,y], transform=fig.transFigure, color="black")
#fig.add_artist(line)
#line = plt.Line2D([0,1],[y+1,y+1], transform=fig.transFigure, color="black")
#fig.add_artist(line)

fig.savefig('exp/convolutionproof.png')


