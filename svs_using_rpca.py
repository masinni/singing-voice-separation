import os

import numpy as np
from numpy.linalg import norm
from hlsvdpro import propack

import soundfile as sf

from datetime import datetime

from stft import stft
from istft import istft

'''
We perform singing voice separation using robust principal component analysis
on a sample of four songs from MIR-1K dataset based on P.-S. Huang, S. D. Chen,
P. Smaragdis, M. Hasegawa-Johnson, "Singing-Voice Separation From Monaural
Recordings Using Robust Principal Component Analysis," in ICASSP 2012.

This is an exact translation of the MATLAB algorithm written by Po-Sen Huang.
'''

def choosvd(n, d):
    y = False
    if n   <= 100: y = d / n <= 0.02
    elif n <= 200: y = d/n <= 0.06
    elif n <= 300: y = d/n <= 0.26
    elif n <= 400: y = d/n <= 0.28
    elif n <= 500: y = d/n <= 0.34
    else: y = d/n <= 0.38
    return y


def inexact_alm_rpca(X, lmbda=None, tol=1e-7, maxiter=1000):
    X = X.T
    Y = X.copy()
    norm_two = np.linalg.svd(Y, full_matrices=False, compute_uv=False)[0]
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])

    Y = Y / dual_norm
    A = np.zeros(Y.shape[1])
    E = np.zeros(Y.shape[1])
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    dnorm = norm(X, 'fro')

    sv = int(10)
    n = Y.shape[1]
    # m = Y.shape[0]
    itr = 0
    total_svd = 0
    stopCriterion = 1
    Converged = False

    while not Converged:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)

        sparse_svd = choosvd(n, sv)
        if sparse_svd:
            U, S, V = propack.svdp(X - Eupdate + (1 / mu) * Y,
                                   sv,
                                   which="L",
                                   kmax=10*sv)
            V = V.T
        else:
            U, S, V = np.linalg.svd(X - Eupdate + (1/mu)*Y,
                                    full_matrices=False)
            V = V.T

        svp = len(np.where(S > (1 / mu))[0])

        if svp < sv:
            sv = int(np.min([svp+1, n]))
        else:
            sv = int(np.min([svp + np.round(0.05 * n), n]))

        Aupdate = np.dot(np.dot(U[:, :svp],
                                np.diag(S[:svp] - 1 / mu)),
                                V[:, :svp].T)

        total_svd += 1
        A = Aupdate
        E = Eupdate

        Z = X - A - E
        Y = Y + mu * Z
        mu = np.minimum(mu * rho, mu_bar)
        itr += 1

        stopCriterion = np.linalg.norm(Z, 'fro') / dnorm

        if stopCriterion <= tol:
            print('StopCriterion < tol', stopCriterion)
            Converged = True
            break

        if Converged == 0 and itr >= maxiter:
            print('Maximum iterations reached')
            Converged = True

    return A, E


def singing_voice_separation(mixture_dir,
                             mixture_filepath,
                             results_dir,
                             vocals,
                             background,
                             lmbda,
                             nFFT,
                             h,
                             gain,
                             power,
                             mask=False):

    filepath = mixture_dir+mixture_filepath
    vocals_filepath = vocals+'/'+mixture_filepath
    background_filepath = background+'/'+mixture_filepath

    # Separate mix:
    data, sr = sf.read(filepath)

    scf = 2 / 3.0
    S_mix = scf * stft(data, f=nFFT, w=nFFT, h=h)

    try:
        A_mag, E_mag = inexact_alm_rpca(np.power(np.abs(S_mix), power),
                                        lmbda / np.sqrt(max(S_mix.shape)))
        PHASE = np.angle(S_mix.conj().T)
    except ValueError:
        print('VALUE ERROR')
        A_mag, E_mag = inexact_alm_rpca(np.power(np.abs(S_mix), power).T,
                                        lmbda / np.sqrt(max(S_mix.shape)))
        PHASE = np.angle(S_mix)

    A = A_mag * np.exp(1j * PHASE)  # lowrank
    E = E_mag * np.exp(1j * PHASE)  # sparse

    if mask:
        mask = np.abs(E) - 1.0 * np.abs(A)
        mask = (mask > 0) * 1
        try:
            Emask = np.multiply(mask, S_mix)
            Amask = np.multiply(1 - mask, S_mix)
        except ValueError:
            Emask = np.multiply(mask, S_mix.conj().T)
            Amask = np.multiply(1 - mask, S_mix.conj().T)
    else:
        Emask = E
        Amask = A

    try:
        wavoutA = istft(Amask.conj().T, ftsize=nFFT, w=nFFT, h=h).conj().T
        wavoutE = istft(Emask.conj().T, ftsize=nFFT, w=nFFT, h=h).conj().T
    except ValueError:
        print('Value Error in ISTFT')
        wavoutA = istft(Amask, ftsize=nFFT, w=nFFT, h=h).conj().T
        wavoutE = istft(Emask, ftsize=nFFT, w=nFFT, h=h).conj().T

    # normalization
    wavoutE /= np.abs(wavoutE).max()
    wavoutA /= np.abs(wavoutA).max()

    # write separated files in wav
    sf.write(vocals_filepath, wavoutE, sr)  # vocals
    sf.write(background_filepath, wavoutA, sr)  # accompaniment


def run_separation(wavfiles, lmbda):

    print('l= ', lmbda)
    results_dir = f'rpca_results/no_mask/results_for_l={lmbda}/'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    vocals = f'rpca_results/no_mask/results_for_l={lmbda}/vocals'
    if not os.path.isdir(vocals):
        os.mkdir(vocals)

    background = f'rpca_results/no_mask/results_for_l={lmbda}/background'
    if not os.path.isdir(background):
        os.mkdir(background)

    start = datetime.now()
    for wav in os.listdir(wavfiles):
        print(wav)
        singing_voice_separation(wavfiles,
                                 wav,
                                 results_dir,
                                 vocals,
                                 background,
                                 lmbda=lmbda,
                                 nFFT=1024,
                                 h=256,
                                 gain=1.0,
                                 power=1.0,
                                 mask=False)

    difference = datetime.now() - start
    print(difference)


if __name__ == '__main__':

    lmbda_list = [1.0]  # [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    wavfiles = 'sample/original_sources/orig_mono/'

    # Separation:
    for lmbda in lmbda_list:
        run_separation(wavfiles, lmbda)
