import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def generate_mel_spectrogram(file_path, sr=16000, n_mels=128, hop_length=512, fixed_length=64):
        audio, _ = librosa.load(file_path, sr=sr)
        
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        if log_mel_spectrogram.shape[1] < fixed_length:
            padding = fixed_length - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')
        else:
            log_mel_spectrogram = log_mel_spectrogram[:, :fixed_length]
        
        return log_mel_spectrogram


file_path = "DATA/ben/ben-50.wav"
mel_spectrogram = generate_mel_spectrogram(file_path)


plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram, sr=16000, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show()
