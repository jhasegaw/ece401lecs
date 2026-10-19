import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('exp', exist_ok=True)

def resonator(x, F, BW, Fs):
    C = -np.exp(-2*np.pi*BW/Fs)
    B = 2 * np.exp(-np.pi*BW/Fs) * np.cos(2*np.pi*F/Fs)
    A = 1 - B - C
    y = np.zeros(len(x))
    y[0] = A*x[0]
    y[1] = A*x[1] + B*y[0]
    for n in range(2,len(y)):
        y[n] = A*x[n] + B*y[n-1] + C*y[n-2]
    return y


Fs = 8000
F0 = 110
T0 = int(np.round(Fs/F0))  # rounding will make T0=73

excitation = np.zeros(8000)
excitation[::T0] = -1   # start=0, end=end of array, step=T0 

y1 = resonator(excitation, 300, 80, Fs)
y2 = resonator(y1, 900, 180, Fs)
y3 = resonator(y2, 2500, 300, Fs)
speech = resonator(y3, 3500, 400, Fs)

x = speech[int(T0/2):int(4.5*T0)]
raw_autocorr = np.correlate(x, x, mode='full')
r = raw_autocorr[raw_autocorr.size // 2:]

t = np.arange(len(x)) / T0

fig = plt.figure(figsize=(14,8),layout='tight')
axs = fig.subplots(3,1)

axs[0].plot(t,x)
axs[0].set_title('$x[n]$: Four pitch periods from a synthetic speech signal',fontsize=18)
axs[0].set_xticks([0,1,2,3])
axs[0].set_xticklabels(['0','$T_0$','$2T_0$','$3T_0$'],fontsize=14)
axs[1].plot(t,r)
axs[1].set_title('$r_x[n]$: Autocorrelation of the speech signal',fontsize=18)
axs[1].set_xticks([0,1,2,3])
axs[1].set_xticklabels(['0','$T_0$','$2T_0$','$3T_0$'],fontsize=14)

R = np.abs(np.fft.fft(x))**2
N_over_T0 = np.arange(len(R))*T0/len(R)
axs[2].plot(N_over_T0[:int(10*len(R)/T0)],R[:int(10*len(R)/T0)])
axs[2].set_title('$|X(\omega)|^2$: Power spectrum of speech signal',fontsize=18)
axs[2].set_xticks(np.arange(9))
axs[2].set_xticklabels(['0','$\omega_0$']+['$%d\omega_0$'%(d) for d in range(2,9)],fontsize=14)

fig.savefig('exp/speech_and_autocorrelation.png')

