import numpy as np
import matplotlib.pyplot as plt
import librosa.display

# Extract spectrogram for mixed signal
scale, sr = librosa.load('sample/original_sources/ORIG_mono/yifen_2_11.wav')
FRAME_SIZE = 1024
HOP_SIZE = 256
S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scale = np.abs(S_scale) ** 2
Y_log_scale = librosa.power_to_db(Y_scale)

# Extract spectrogram for singing voice
scalev, sr = librosa.load('sample/separated_rpca/results_for_l=1.0/vocals/yifen_2_11.wav')
FRAME_SIZE = 1024
HOP_SIZE = 256
S_scalev = librosa.stft(scalev, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scalev = np.abs(S_scalev) ** 2
Y_log_scale_v = librosa.power_to_db(Y_scalev)

# Extract spectrogram fro music accompaniment
scaleb, sr = librosa.load('sample/separated_rpca/results_for_l=1.0/background/yifen_2_11.wav')
FRAME_SIZE = 1024
HOP_SIZE = 256
S_scaleb = librosa.stft(scaleb, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
Y_scaleb = np.abs(S_scaleb) ** 2
Y_log_scale_b = librosa.power_to_db(Y_scaleb)


def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

# Plot spectrograms
sr = 16000
plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")
plot_spectrogram(Y_log_scale_v, sr, HOP_SIZE, y_axis="log")
plot_spectrogram(Y_log_scale_b, sr, HOP_SIZE, y_axis="log")