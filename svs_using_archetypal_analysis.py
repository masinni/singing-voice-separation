"""
We perform singing voice separation using Archetypal Analysis with sparseness
constraints. We use the same stft, istft and evaluation functions as in SVS
with RPCA in order to compare both methods.
"""

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
    The number of archetypes, k, is also the rank of matrices C,S.

    rpca_rank == True:
    we consider the rank of the low rank matrix seperated with archetypal
    analyis method equal to the rank of the low-rank matrix seperated with
    RPCA method (for comparison reasons).
    We have saved the singular-values matrix S, taken from the Singular
    Value Decomposition that we applyed to the low-rank matrix A separated
    using RPCA. Matrix S is equal to the rank of a matrix by applying
    a threshold to the values that are almost zero.

    Parameters:
    - lmbda: Regularization parameter.
    - mixture_filepath: Filepath of the mixture.
    - rpca_rank: Boolean indicating whether to consider RPCA rank or use a default value.

    Returns:
    - k: Number of archetypes.
    """
    if rpca_rank:
        # Compute rank of low-rank matrix separated with RPCA
        lmbda_values = {0.1: 0.1, 0.5: 1.0, 1.0: 10.0, 1.5: 10.0, 2.0: 15.0, 2.5: 15.0}
        sing_val_path = f'rpca_results/no_mask/results_for_l={lmbda}/matrices/singular_values_of_A/'

        if lmbda not in lmbda_values:
            lmbda = 1.0  # Default value if lmbda is not in the predefined values

        track = os.path.join(sing_val_path, f'S_{mixture_filepath}.npy')
        sing_A = np.load(track)

        threshold = lmbda_values[lmbda]
        super_threshold_indices = sing_A < threshold
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

    filepath = os.path.join(mixture_dir, mixture_filepath)
    vocals_filepath = os.path.join(vocals, mixture_filepath)
    background_filepath = os.path.join(background, mixture_filepath)

    k = number_of_archetypes(lmbda, mixture_filepath, rpca_rank)
    print('number of archetypes = ', k)

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

    # Evaluation:
    archet_eval = evaluation.Eval()
    archet_eval.eval_metrics(wavfiles, lmbda_list)
