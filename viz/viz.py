
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

import torchaudio.transforms as T
import librosa 

import numpy as np


def audio_spectrogram_image(waveform, power=2.0, sample_rate=48000):
    """
    # cf. https://pytorch.org/tutorials/beginner/audio_feature_extractions_tutorial.html
    """
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 80

    mel_spectrogram_op = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, win_length=win_length, 
        hop_length=hop_length, center=True, pad_mode="reflect", power=power, 
        norm='slaney', onesided=True, n_mels=n_mels, mel_scale="htk")

    melspec = mel_spectrogram_op(waveform.float())
    melspec = melspec[0] # TODO: only left channel for now

    fig = Figure(figsize=(12, 4))
    canvas = FigureCanvasAgg(fig)
    axs = fig.add_subplot()
    librosa.display.specshow(librosa.power_to_db(melspec), ax=axs, y_axis='mel', fmax=sample_rate//2)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, 0:3]
