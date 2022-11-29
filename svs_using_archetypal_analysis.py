import os
import numpy as np
import soundfile as sf
from datetime import datetime

import class_archetypal_analysis
from stft import stft
from istft import istft
import evaluation


def number_of_archetypes(lmbda, mixture_filepath, rpca_rank):
    """
    k = rank of low matrix.
    for comparison reasons we consider the rank of the low rank matrix
    seperated with archetypal analyis method equal to the low-rank matrix
    seperated with RPCA method.

    we compute the rank of each track individually, by 
    """
    if rpca_rank:
        # Compute rank of low-rank matrix separated with RPCA
        if lmbda == 0.1 or lmbda == 0.5 or lmbda == 2.0 or lmbda == 2.5:
            sing_val = \
                f'rpca_results/no_mask/results_for_l={lmbda}/matrices/singular_values_of_A/'
        else:
            sing_val = \
                'rpca_results/no_mask/results_for_l=1.0/matrices/singular_values_of_A/'

        track = sing_val + f'S_{mixture_filepath}.npy'

        sing_A = np.load(track)

        if lmbda == 0.1:
            super_threshold_indices = sing_A < 1.0
        elif lmbda == 0.5:
            super_threshold_indices = sing_A < 1.0
        elif lmbda == 1.0 or 1.5:
            super_threshold_indices = sing_A < 10.0
        elif lmbda == 2.0 or 2.5:
            super_threshold_indices = sing_A < 15.0
        else:
            super_threshold_indices = sing_A < 25.0

        sing_A[super_threshold_indices] = 0
        k = int(np.count_nonzero(sing_A))
    else:
        k = 10

    return k


def singing_voice_separation(mixture_dir,
                             mixture_filepath,
                             results_dir,
                             vocals,
                             background,
                             lmbda,
                             nFFT,
                             h,
                             rpca_rank):

    filepath = mixture_dir+mixture_filepath
    vocals_filepath = vocals+'/'+mixture_filepath
    background_filepath = background+'/'+mixture_filepath

    k = number_of_archetypes(lmbda, mixture_filepath, rpca_rank)
    print('number of archetypes=', k)
    # Separate mix:
    sr = 16000
    data, sr = sf.read(filepath)

    scf = 2 / 3.0
    S_mix = scf * stft(data, f=nFFT, w=nFFT, h=h)

    archet = class_archetypal_analysis.Archetypal_analysis_sparseness(
                X=np.abs(S_mix),
                k=k,
                lmbda=lmbda,
                max_iter=700,
                wav_name=mixture_filepath)
    A_mag, E_mag = archet.matrix_formulation()
    PHASE = np.angle(S_mix.conj().T)

    A = A_mag.T * np.exp(1j * PHASE)  # lowrank
    E = E_mag.T * np.exp(1j * PHASE)  # sparse

    wavoutA = istft(A.conj().T, ftsize=nFFT, w=nFFT, h=h).conj().T
    wavoutE = istft(E.conj().T, ftsize=nFFT, w=nFFT, h=h).conj().T

    # normalization
    wavoutE /= np.abs(wavoutE).max()
    wavoutA /= np.abs(wavoutA).max()

    sf.write(vocals_filepath, wavoutE, sr)  # vocals
    sf.write(background_filepath, wavoutA, sr)  # accompaniment


def run_separation(wavfiles, lmbda):

    print('l= ', lmbda)
    results_dir = f'archet_analysis_results/results_for_l={lmbda}/'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    vocals = f'archet_analysis_results/results_for_l={lmbda}/vocals'
    if not os.path.isdir(vocals):
        os.mkdir(vocals)

    background = f'archet_analysis_results/results_for_l={lmbda}/background'
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
                                 rpca_rank=True)

    difference = datetime.now() - start
    print(difference)


if __name__ == '__main__':

    lmbda_list = [1.0]  # [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    wavfiles = 'sample/original_sources/orig_mono/'

    # Separation:
    for lmbda in lmbda_list:
        run_separation(wavfiles, lmbda)

    # Evaluation metrics:
    test = evaluation.Eval()
    med_sdr_list2, med_sir_list2, med_sar_list2 = test.eval_metrics(
                                                  wavfiles,
                                                  lmbda_list)
