import numpy as np
import math

'''
d = stft(x, f , w, h) Short-time Fourier transform.
Returns some frames of short-term Fourier transform of x.
Each column of the result is one f-point fft each
successive frame is offset by h points (w/2) until X is exhausted.
'''


def stft(x, f, w, h):

    # expect x as a row
    if x.shape[0] > 1:
        x = x[np.newaxis, :]
    s = max(x.shape)

    if np.fmod(w, 2) == 0:
        w = w + 1
    halflen = (w-1)//2
    halff = f//2
    clmn = np.arange(halflen+1)[np.newaxis, :]
    den = 1+np.cos(np.multiply(math.pi, np.divide(clmn, halflen)))
    halfwin = np.multiply(0.5, den)
    win = np.zeros((1, f))
    acthalflen = np.minimum(halff, halflen)
    j = 0
    for i in range(halff, halff+acthalflen):
        win[0, i] = halfwin[0, 0+j]
        j += 1
        if j == acthalflen+1:
            break
    j = 0
    for i in range(halff, halff-acthalflen, -1):
        win[0, i] = halfwin[0, 0+j]
        j += 1
        if j == acthalflen+1:
            break

    w = max(win.shape)
    c = 0

    # pre-allocate output array
    d = np.zeros([int(1 + f/2), int(1+np.fix((s-f)/h))], dtype='complex_')

    for b in range(0, s-f+1, h):
        u = win * x[0, b:b+f]
        t = np.fft.fft(u)
        d[:, c] = t[0, 0:int((1+f/2))].conj().T
        c += 1

    return d
