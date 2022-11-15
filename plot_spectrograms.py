"""
Visualization of magnitude spectrograms using librosa and matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa.display


def song_to_spectro(frame_size, hop_size, mix, vocals, music):
    # Extract spectrogram for mixed signal
    scale, sr = librosa.load(mix)
    S_scale = librosa.stft(scale, n_fft=frame_size, hop_length=hop_size)
    Y_scale = np.abs(S_scale) ** 2
    Y_log_scale = librosa.power_to_db(Y_scale)

    # Extract spectrogram for singing voice
    scalev, sr = librosa.load(vocals)
    S_scalev = librosa.stft(scalev, n_fft=frame_size, hop_length=hop_size)
    Y_scalev = np.abs(S_scalev) ** 2
    Y_log_scale_v = librosa.power_to_db(Y_scalev)

    # Extract spectrogram for music accompaniment
    scaleb, sr = librosa.load(music)
    S_scaleb = librosa.stft(scaleb, n_fft=frame_size, hop_length=hop_size)
    Y_scaleb = np.abs(S_scaleb) ** 2
    Y_log_scale_b = librosa.power_to_db(Y_scaleb)

    return Y_log_scale, Y_log_scale_v, Y_log_scale_b


def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")


if __name__ == '__main__':

    # Choose wav file
    wav_file = "yifen_2_11.wav"

    # Load paths
    mix = f'sample/original_sources/orig_mono/{wav_file}'
    vocals = f'sample/separated_rpca/results_for_l=1.0/vocals/{wav_file}'
    music = f'sample/separated_rpca/results_for_l=1.0/background/{wav_file}'

    # Choose frame size and hop size
    FRAME_SIZE = 1024
    HOP_SIZE = 256

    # Extract log power spectrograms
    Y_log_scale, Y_log_scale_v, Y_log_scale_b = song_to_spectro(FRAME_SIZE,
                                                                HOP_SIZE,
                                                                mix,
                                                                vocals,
                                                                music)

    # Plot spectrograms
    sr = 16000
    plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
    plot_spectrogram(Y_log_scale_v, sr, HOP_SIZE, y_axis="log")
    plot_spectrogram(Y_log_scale_b, sr, HOP_SIZE, y_axis="log")
