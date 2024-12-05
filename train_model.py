import os
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
import numpy as np

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

def train_model(model, train_loader, val_loader, epochs=10):
    best_val_acc = 0.0
    best_model_path = "best_model.pth"
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
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Acc: {best_val_acc:.2f}%")

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")
    
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_dir = "DATA/"
X = []
Y = []
for idx, class_folder in enumerate(os.listdir(train_dir)):
    class_folder_path = os.path.join(train_dir, class_folder)
    if os.path.isdir(class_folder_path):
        y = CLASS_TO_LABEL[class_folder]
        for sample in os.listdir(class_folder_path):
            if sample.endswith('.wav'):
                file_path = os.path.join(class_folder_path, sample)
                X.append(file_path)
                Y.append(y)
X = np.array(X)
Y = np.array(Y)

val_dir = "EVAL/"
val_X = []
val_Y = []
for idx, class_folder in enumerate(os.listdir(val_dir)):
    class_folder_path = os.path.join(val_dir, class_folder)
    if os.path.isdir(class_folder_path):
        y = CLASS_TO_LABEL[class_folder]
        for sample in os.listdir(class_folder_path):
            if sample.endswith('.wav'):
                file_path = os.path.join(class_folder_path, sample)
                val_X.append(file_path)
                val_Y.append(y)
val_X = np.array(val_X)
val_Y = np.array(val_Y)
transform = transform_audio()

train_dataset = AudioDataset(X,Y, transform=transform)
val_dataset = AudioDataset(val_X,val_Y, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32,collate_fn=collate_fn)
num_classes = 7
model = AudioClassifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

train_model(model, train_loader, val_loader, epochs=50)

