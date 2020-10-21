import numpy as np
import matplotlib.figure, subprocess, os, wave

os.makedirs('exp',exist_ok=True)


################################################################################
# Picture  showing glottal input, filter, speech output.
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)
N = 400
n0 = [ 45+80*m for m in range(5) ]
G = 7.2
delta = np.zeros(N)
delta[0] = 1
F = np.array([800, 1200, 2800, 3200])
B = np.array([100, 100, 400, 600])
fs = 8000
p = np.exp(-np.pi*B/fs)*np.exp(2j*np.pi*F/fs)
h = delta.copy()
for k in range(4):
    h[1] = h[1] + 2*np.real(p[k])*h[0]
    for n in range(2,N):
        h[n] = h[n] + 2*np.real(p[k])*h[n-1] - np.square(np.abs(p[k]))*h[n-2]
x = np.zeros(N)
y = np.zeros(N)
for m in range(5):
    x -= G*np.concatenate((delta[N-n0[m]:N],delta[:N-n0[m]]))
    y -= G*np.concatenate((h[N-n0[m]:N],h[:N-n0[m]]))
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Air pressure at glottis = series of negative impulses')
axs[1].plot(h)
axs[1].set_title('Impulse response of the vocal tract = damped resonances')
axs[2].plot(y)
axs[2].set_title('Air pressure at lips = series of damped resonances')
axs[2].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speech_fivepulses.png')

################################################################################
# Picture showing impulse input, speech output.
fig = matplotlib.figure.Figure((6,4))
axs = fig.subplots(3,1,sharex=True)
N = 80
n0 = 45
delta = np.zeros(N)
delta[0] = 1
h = delta.copy()
for k in range(4):
    h[1] = h[1] + 2*np.real(p[k])*h[0]
    for n in range(2,N):
        h[n] = h[n] + 2*np.real(p[k])*h[n-1] - np.square(np.abs(p[k]))*h[n-2]
x = -G*np.concatenate((delta[N-n0:N],delta[:N-n0]))
y = -G*np.concatenate((h[N-n0:N],h[:N-n0]))
axs[0].stem(x,use_line_collection=True)
axs[0].set_title('Air pressure at glottis = $G\delta[n-n_0]$, once per frame')
axs[1].plot(h)
axs[1].set_title('Impulse response of the vocal tract')
axs[2].plot(y)
axs[2].set_title('Air pressure at lips = $Gh[n-n_0]$, once per frame')
axs[2].set_xlabel('Time (samples)')
fig.tight_layout()
fig.savefig('exp/speech_onepulse.png')

################################################################################
# Picture showing the inverse filtering result, maybe from a real speech signal.
with wave.open('../lec15/ow.wav', 'rb') as f:
    nsamples = f.getnframes()
    wav = np.frombuffer(f.readframes(nsamples),dtype=np.int16).astype('float32')
    ow = wav[int(0.5*len(wav)-200):int(0.5*len(wav)+201)]
    ow /= np.amax(np.abs(ow))

r = np.correlate(ow,ow,'full')
print(r[398:403])
print(len(r))
r = r[int((len(r)-1)/2):]
print(len(r))
R = np.diag(np.tile(r[0],10))
for m in range(1,10):
    R += np.diag(np.tile(r[m],10-m),m)
    R += np.diag(np.tile(r[m],10-m),-m)
gamma = r[1:11]
a = np.matmul(np.linalg.inv(R),gamma.reshape((10,1)))
e = np.zeros(ow.shape)
for n in range(len(ow)):
    e[n] = ow[n]
    for m in range(min(n,10)):
        e[n] -= a[m]*ow[n-(m+1)]
fig = matplotlib.figure.Figure((10,4))
axs = fig.subplots(3,1)
t = np.arange(len(ow))/16000
axs[0].plot(t,ow)
axs[0].set_title('Waveform, $s[n]$, of the vowel /o/')
axs[1].stem(np.arange(1,11),a,use_line_collection=True)
axs[1].set_title('Predictor Coefficients $a_k$')
axs[2].plot(t[11:],e[11:])
axs[2].set_title('Result of Inverse Filtering, $e[n]=s[n]-sum_k a_k s[n-k]$')
axs[2].set_xlabel('Time (sec)')
fig.tight_layout()
fig.savefig('exp/inversefilter.png')
