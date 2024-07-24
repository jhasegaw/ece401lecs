import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs('exp',exist_ok=True)

#########################################################################################
# 4-sample-signal
fig, ax = plt.subplots(figsize=(14,4))
ax.stem(np.arange(0,4), np.array([1,1,0,0]))
ax.set_xlim(-0.5,3.5)
ax.set_ylim(-0.5,1.5)
ax.set_title('x[n] = a very simple four-sample signal')
fig.tight_layout()
fig.savefig('exp/simple_signal.png')


#########################################################################################
# dft_of_cosine
N = 64
omega1 = 2*np.pi*20.3/N
signal1 = np.cos(omega1*np.arange(N))
zero_padded_signal1 = np.concatenate((np.zeros(N),np.zeros(N),signal1,np.zeros(N),np.zeros(N)))

omega = np.linspace(0,2*np.pi,5*N,endpoint=False)
omega_k = np.linspace(0,2*np.pi, N, endpoint=False)
absDTFT1 = np.abs(np.fft.fft(zero_padded_signal1))
absDFT1 = np.abs(np.fft.fft(signal1))
fig,axs = plt.subplots(2,1,figsize=(14,8))
axs[0].plot(omega,absDTFT1)
axs[0].stem(omega_k,absDFT1)
axs[0].set_title('Magnitude DTFT and DFT of cos(2π 20.3/N)')
axs[1].stem(omega_k,absDFT1)
axs[1].set_title('Magnitude DFT of cos(2π 20.3/N)')
axs[1].set_xlabel('Frequency (radians/sample)')
fig.tight_layout()
fig.savefig('exp/dft_of_cosine1.png')

omega2 = 2*np.pi*20/N
signal2 = np.cos(omega2*np.arange(N))
zero_padded_signal2 = np.concatenate((np.zeros(N),np.zeros(N),signal2,np.zeros(N),np.zeros(N)))

absDTFT2 = np.abs(np.fft.fft(zero_padded_signal2))
absDFT2 = np.abs(np.fft.fft(signal2))
fig,axs = plt.subplots(2,1,figsize=(14,8))
axs[0].plot(omega,absDTFT2)
axs[0].stem(omega_k,absDFT2)
axs[0].set_title('Magnitude DTFT and DFT of cos(2π 20/N)')
axs[1].stem(omega_k,absDFT2)
axs[1].set_title('Magnitude DFT of cos(2π 20/N)')
axs[1].set_xlabel('Frequency (radians/sample)')
fig.tight_layout()
fig.savefig('exp/dft_of_cosine2.png')
