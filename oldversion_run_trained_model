def run_trained_model(X):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torchaudio
    import gdown
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Compose

    class AudioClassifier(nn.Module):
        def __init__(self, num_classes):
            super(AudioClassifier, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_pool(x)
            x = self.fc(x)
            return x
    
    class AudioDataset(Dataset):
        def __init__(self, filepaths, labels, transform=None):
            self.filepaths = filepaths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            filepath = self.filepaths[idx]
            label = self.labels[idx]

            waveform, _ = torchaudio.load(filepath)
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if self.transform:
                waveform = self.transform(waveform)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            return waveform, label
            
    def transform_audio(sample_rate=16000, n_fft=512, n_mels=64, hop_length=256):
        return Compose([
            torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            ),
            torchaudio.transforms.AmplitudeToDB()
        ])

    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        max_len = max(waveform.shape[-1] for waveform in waveforms)
        padded_waveforms = [torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[-1])) for waveform in waveforms]
        return torch.stack(padded_waveforms), torch.tensor(labels, dtype=torch.long)
    
    def download_weights(url, output_path):
        gdown.download(url, output_path, quiet=False)
        print("Weights downloaded.")

    def load_weights_to_new_model(trained_weights_path, num_classes, device):
        model = AudioClassifier(num_classes=num_classes).to(device)
        state_dict = torch.load(trained_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def classifier(X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_url = "https://drive.google.com/uc?id=1iULKy5DLiNLoPkqmcW8JNY5gtO6gfeFy"
        weights_path = "my_weights.pth"
        download_weights(weights_url, weights_path)
        model = load_weights_to_new_model(weights_path, 7, device)
        transform = transform_audio()
        val_dataset = AudioDataset(X, Y, transform=transform)
        eval_loader = DataLoader(val_dataset, batch_size=32,collate_fn=collate_fn)
        all_preds = []
        with torch.no_grad():
            for batch_waveforms, _ in eval_loader:
                batch_waveforms = batch_waveforms.to(device)
                outputs = model(batch_waveforms)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)
    predictions = classifier(X)
    assert predictions.shape == Y.shape
    return predictions
