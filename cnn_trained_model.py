def run_trained_model(X):
    import os
    import numpy as np
    import librosa
    import librosa.display
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.model_selection import StratifiedKFold
    from torch.optim import Adam
    import gdown

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
    
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths, labels, transform=None):
            self.file_paths = file_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.file_paths)
        
        def __getitem__(self, idx):
            file_path = self.file_paths[idx]
            label = self.labels[idx]

            spectrogram = generate_mel_spectrogram(file_path)
            
            if self.transform:
                spectrogram = self.transform(spectrogram)

            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(label, dtype=torch.long)
            
            return spectrogram, label
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(26880, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        
    def download_weights(url, output_path):
        gdown.download(url, output_path, quiet=False)
        print("Weights downloaded.")

    def load_weights_to_new_model(trained_weights_path):
        model = SimpleCNN(num_classes=7)
        state_dict = torch.load(trained_weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def classifier(X):
        weights_url = "https://drive.google.com/uc?id=1CzJiWhlARalfZ5AbM5RUJcTwsx2Jy974"
        weights_path = "my_weights_cnn.pth"
        download_weights(weights_url, weights_path)
        model = load_weights_to_new_model(weights_path)
        val_dataset = AudioDataset(X, Y)
        eval_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for spectrograms, labels in eval_loader:
                outputs = model(spectrograms)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())

        return np.array(all_preds)
    predictions = classifier(X)
    assert predictions.shape == Y.shape
    return predictions


import os
import numpy as np
CLASS_TO_LABEL = {
    'water': 0,
    'table': 1,
    'sofa': 2,
    'railing': 3,
    'glass': 4,
    'blackboard': 5,
    'ben': 6,
}

LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

base_dir = 'DATA'
X = []
Y = []
for idx, class_folder in enumerate(os.listdir(base_dir)):
    class_folder_path = os.path.join(base_dir, class_folder)
    if os.path.isdir(class_folder_path):
        y = CLASS_TO_LABEL[class_folder]
        for sample in os.listdir(class_folder_path):
            file_path = os.path.join(class_folder_path, sample)
            X.append(file_path)
            Y.append(y)
X = np.array(X)
Y = np.array(Y)

from sklearn.metrics import accuracy_score

Y_pred = run_trained_model(X)
accuracy = accuracy_score(Y, Y_pred)
print(f"Model Accuracy on unseen data: {accuracy:.2f}")