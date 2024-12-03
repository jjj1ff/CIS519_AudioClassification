import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            transform (callable, optional): A function/transform to apply to the audio data.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
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
        waveform, sample_rate = torchaudio.load(self.filepaths[idx])
        if self.transform:
            waveform = self.transform(waveform)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        print(f"Input shape: {waveform.shape}")
        return waveform, self.labels[idx]


def transform_audio(sample_rate=16000, n_fft=400, n_mels=64):
    return Compose([
        torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000),
        torchaudio.transforms.MelSpectrogram(n_fft=n_fft, n_mels=n_mels),
        torchaudio.transforms.AmplitudeToDB()
    ])

train_dir = "C:/Users/asus/Downloads/CIS519-proj-20241203T182027Z-001/DATA"
val_dir = "C:/Users/asus/Downloads/Sample_Evaluation_Data-20241203T185510Z-001/Sample_Evaluation_Data"
transform = transform_audio()

train_dataset = AudioDataset(root_dir=train_dir, transform=transform)
val_dataset = AudioDataset(root_dir=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


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
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

num_classes = 7
model = AudioClassifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

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
    return val_loss / len(val_loader), val_acc

train_model(model, train_loader, val_loader, epochs=10)