import numpy as np
import matplotlib.figure, subprocess, os, wave

###########################################################################
# Plots showing waveforms of /i/ and /a/, time domains in milliseconds and in samples.
with wave.open('aa.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    aa = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    aa /= np.amax(np.abs(aa))
with wave.open('iy.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    iy = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    iy /= np.amax(np.abs(iy))
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
t = np.arange(len(aa))/16000
axs[0,0].plot(t,aa)
axs[0,0].set_title('Waveform of the vowel /a/, $F_1=800$Hz')
axs[0,0].set_xlabel('Time (sec)')
axs[1,0].plot(t,iy)
axs[1,0].set_title('Waveform of the vowel /i/, $F_1=400$Hz')
axs[1,0].set_xlabel('Time (sec)')
axs[0,1].plot(aa)
axs[0,1].set_title('Waveform of the vowel /a/, $\omega_1=π/10$')
axs[0,1].set_xlabel('Time (samples)')
axs[1,1].plot(iy)
axs[1,1].set_title('Waveform of the vowel /i/, $\omega_1=π/20$')
axs[1,1].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speechwaves.png')

###########################################################################
# Plots showing spectra of /i/ and /a/, freq domains in Hertz and in radians/sample.
L = int(0.015*16000)
w = np.hamming(L)
print(np.amin(w),np.amax(w))
print(np.amin(aa),np.amax(aa))
N = 8192
AA = np.zeros(N)
IY = np.zeros(N)
for n in range(0,len(aa)-L,int(L/2)):
    AA += np.square(np.abs(np.fft.fft(aa[n:(n+L)]*w, n=N)))
    IY += np.square(np.abs(np.fft.fft(iy[n:(n+L)]*w, n=N)))
print(np.amin(AA),np.amax(AA))
AA = np.sqrt(AA)
IY = np.sqrt(IY)
print(np.amin(AA),np.amax(AA))
f = np.linspace(0,16000,N,endpoint=False)
omega = 2*np.pi*f/16000
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(2,2)
axs[0,0].plot(f[f<3500],20*np.log10(AA[f<3500]))
axs[0,0].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /a/, $F_1=800$Hz')
axs[0,0].set_xlabel('Freq (Hz)')
axs[1,0].plot(f[f<3500],20*np.log10(IY[f<3500]))
axs[1,0].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /i/, $F_1=400$Hz')
axs[1,0].set_xlabel('Freq (Hz)')
axs[0,1].plot(omega[f<3500],20*np.log10(AA[f<3500]))
axs[0,1].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /a/, $\omega_1=π/10$')
axs[0,1].set_xlabel('Freq (radians/sample)')
axs[1,1].plot(omega[f<3500],20*np.log10(IY[f<3500]))
axs[1,1].set_title('Estimated $20log_{10}|H(\omega)|$ of the vowel /i/, $\omega_1=π/20$')
axs[1,1].set_xlabel('Freq (radians/sample)')
fig.tight_layout()
fig.savefig('exp/speechspecs.png')


