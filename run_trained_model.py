import os
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
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.filepaths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".wav"):
                    self.filepaths.append(os.path.join(class_dir, file_name))
                    self.labels.append(idx)     

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(filepath)

        if self.transform:
            waveform = self.transform(waveform)

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

def evaluate_model(model, val_loader):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.0 * correct / total
    print(f'Loss: {val_loss / len(val_loader)}, Acc: {val_acc}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_url = "https://drive.google.com/uc?id=1iULKy5DLiNLoPkqmcW8JNY5gtO6gfeFy"
weights_path = "my_weights.pth"
eval_folder = "EVAL"
num_classes = 7
batch_size = 32
download_weights(weights_url, weights_path)
model = load_weights_to_new_model(weights_path, 7, device)

transform = transform_audio()
val_dataset = AudioDataset(root_dir=eval_folder, transform=transform)
eval_loader = DataLoader(val_dataset, batch_size=32,collate_fn=collate_fn)
criterion = nn.CrossEntropyLoss()
evaluate_model(model, eval_loader)