import numpy as np
import math

'''
x = istft(d, f, w, h) Inverse short-time Fourier transform.
Performs overlap-add resynthesis from the short-time Fourier transform
data in d.  Each column of d is taken as the result of an f-point fft,
each successive frame was offset by h points.
Data is hann-windowed at w pts.
w as a vector uses that as window.
'''


def istft(d, ftsize, w, h):
    s = d.shape
    if s[0] != ftsize//2+1:
        print('number of rows should be fftsize/2+1')

    cols = s[1]
    if np.fmod(w, 2) == 0:  # force window to be odd-len
        w = w + 1
    halflen = (w-1)//2
    halff = ftsize//2
    clmn = np.arange(halflen+1)[np.newaxis, :]
    den = 1 + np.cos(np.multiply(math.pi, np.divide(clmn, halflen)))
    halfwin = np.multiply(0.5, den)
    win = np.zeros((1, ftsize))
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
    # Make stft-istft loop be identity for 25% hop
    win = np.multiply(2/3, win)

    w = max(win.shape)
    x = np.zeros((1, ftsize + (cols-1)*h))

    for b in range(0, h*(cols-1)+1, h):
        sel = d[:, b//h].conj()
        ftt = sel[np.newaxis, :]
        ela = sel[(ftsize//2)-1:0:-1][np.newaxis, :]
        ft = np.concatenate((ftt, np.conj(ela)), axis=1)
        px = np.real(np.fft.ifft(ft))
        x[0, (b):(b+ftsize)] = x[0, (b):(b+ftsize)]+px*win

    return x
