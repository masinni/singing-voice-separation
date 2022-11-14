import os

import librosa
import numpy as np
import soundfile as sf


class Eval():

    def __init__(self):
        self.samplerate = 16000

    def bss_proj(self, x, Y):
        """
        Computes the orthogonal projection of x on the subspace
        spanned by the row(s) of Y.
        """
        # Gram matrix of Y
        G = np.dot(Y, Y.T)
        A = np.conj(G)
        B = np.conj(np.dot(Y, x.T))
        coeff = np.linalg.solve(A, B)
        PY_x = np.dot(coeff.T, Y)

        return PY_x

    def bss_decomp_gain(self, se, index, S):

        """se: row vector of length T containing the estimated source,
           index: points which component of S se has to be compared to,
           S: n x T matrix containing the original sources"""

        if se.shape[0] > 1:
            se = np.expand_dims(se, axis=0)
        # Create the space of target source(s)
        target_space = np.expand_dims(S[index, :], axis=0)

        # Create the space of sources
        sources_space = S

        # Target source(s) contribution
        s_target = self.bss_proj(se, target_space)

        # Interferences contribution
        P_S_se = self.bss_proj(se, sources_space)
        e_interf = P_S_se - s_target
        e_artif = se - P_S_se
        return s_target, e_interf, e_artif

    # Compute SDR,SIR,SAR
    def safe_db(self, num, den):

        """
        Properly handle the potential +Inf db SIR, instead of raising a
        RuntimeWarning. Only denominator is checked because the numerator can
        never be 0.
        """
        if den == 0:
            return np.Inf
        return 10 * np.log10(num / den)

    def bss_crit(self, s_true, e_interf, e_artif):
        s_filt = s_true
        sdr = self.safe_db(np.sum(s_filt**2, axis=1),
                           np.sum((e_interf + e_artif)**2, axis=1))
        sir = self.safe_db(np.sum(s_filt**2, axis=1),
                           np.sum(e_interf**2, axis=1))
        sar = self.safe_db(np.sum((s_filt + e_interf)**2, axis=1),
                           np.sum(e_artif**2, axis=1))
        return (sdr, sir, sar)

    def fix_size(self,
                 reference_dir,
                 vocal_dir,
                 background_dir,
                 mix_filepath,
                 voc_filepath,
                 bck_filepath):

        # Load original data
        filepath = reference_dir+mix_filepath
        data, sr = librosa.load(filepath, sr=self.samplerate, mono=False)

        # Load background music
        filepath_bck = background_dir+bck_filepath
        A, sr = sf.read(filepath_bck)

        # Load singing voice
        filepath_voc = vocal_dir+voc_filepath
        E, sr = sf.read(filepath_voc)

        # Fix size of sources
        wavinA = data[0, :]  # original background music
        wavinE = data[1, :]  # original singing voice

        if wavinA.shape == A.shape and wavinE.shape == E.shape:
            orig = np.vstack((wavinE, wavinA))
            sep = np.vstack((E, A))

        elif wavinA.shape > A.shape or wavinE.shape > E.shape:
            minlength = min(A.shape, E.shape)
            minvalue = minlength[0]
            wavinA = wavinA[0:int(minvalue)]
            wavinE = wavinE[0:int(minvalue)]
            orig = np.vstack((wavinE, wavinA))
            sep = np.vstack((E, A))

        return orig, sep

    def eval_metrics(self, wavfiles, lmbda_list):

        original_folder = 'sample/original_sources/orig_stereo/'
        original_mono = wavfiles
        original_voice = 'sample/original_sources/orig_voice/'

        med_sdr_list2 = []
        med_sir_list2 = []
        med_sar_list2 = []

        for lmbda in lmbda_list:
            print("l=", lmbda)
            sep_voc_dir = f'rpca_results/no_mask/results_for_l={lmbda}/vocals/'
            sep_bck_dir = f'rpca_results/no_mask/results_for_l={lmbda}/background/'

            sdr_mixture_list = []
            NSDR_list = []
            durations_list = []

            sdr_list2 = []
            sir_list2 = []
            sar_list2 = []

            mix = os.listdir(wavfiles)
            mix = sorted(mix)
            vocals = os.listdir(sep_voc_dir)
            vocals = sorted(vocals)
            background = os.listdir(sep_bck_dir)
            background = sorted(background)

            while len(mix) != 0:
                mix_file = mix.pop(0)
                print(mix_file)
                v = vocals.pop(0)
                b = background.pop(0)

                # GNSDR computation
                wavinEO, sr = librosa.load(original_voice+mix_file,
                                           sr=self.samplerate)
                mix_mono, sr = librosa.load(original_mono+mix_file,
                                            sr=self.samplerate)
                dur = mix_mono.shape[-1]
                durations_list.append(dur)
                [s_target1, e_interf1, e_artif1] = self.bss_decomp_gain(
                                                   mix_mono[np.newaxis, :],
                                                   0,
                                                   wavinEO[np.newaxis, :])

                [sdr_mixture, sir_mixture, sar_mixture] = self.bss_crit(
                                                          s_target1,
                                                          e_interf1,
                                                          e_artif1)
                sdr_mixture_list.append(sdr_mixture)

                orig, sep = self.fix_size(original_folder,
                                          sep_voc_dir,
                                          sep_bck_dir,
                                          mix_file,
                                          v,
                                          b)

                temp_sdr = []
                temp_sir = []
                temp_sar = []
                for i in range(0, sep.shape[0]):
                    [s_target, e_interf, e_artif] = self.bss_decomp_gain(
                                                    sep[i, :],
                                                    i,
                                                    orig)
                    (sdr, sir, sar) = self.bss_crit(s_target,
                                                    e_interf,
                                                    e_artif)
                    temp_sdr.append(sdr)
                    temp_sir.append(sir)
                    temp_sar.append(sar)

                print('VOICE:  sdr =', temp_sdr[0],
                      'sir =', temp_sir[0], 'sar =', temp_sar[0])

                sdr_list2.append(temp_sdr[0])
                sir_list2.append(temp_sir[0])
                sar_list2.append(temp_sar[0])

        # Create list of median of metrics
        med_sdr_list2.append(np.mean(sdr_list2))
        med_sir_list2.append(np.mean(sir_list2))
        med_sar_list2.append(np.mean(sar_list2))

        # NSDR = SDR(estimated voice, voice) - SDR(mixture, voice)
        for i in range(len(sdr_mixture_list)):
            NSDR = float(sdr_list2[i] - sdr_mixture_list[i])
            NSDR_list.append(NSDR)

        # The GNSDR is calculated by taking the mean of the NSDRs over all
        # mixtures of each set, weighted by their length.
        dur_list_sum = sum(durations_list)
        arithm = np.multiply(durations_list, NSDR_list)
        GNSDR = np.divide(np.sum(arithm), dur_list_sum)
        print('GNSDR (dB)', GNSDR)

        return med_sdr_list2, med_sir_list2, med_sar_list2
