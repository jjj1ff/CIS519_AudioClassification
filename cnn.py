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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_val_accuracy = 0.0
best_fold = -1

for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y)):
    train_paths = [X[i] for i in train_idx]
    train_labels = [Y[i] for i in train_idx]
    val_paths = [X[i] for i in val_idx]
    val_labels = [Y[i] for i in val_idx]
    
    train_dataset = AudioDataset(train_paths, train_labels)
    val_dataset = AudioDataset(val_paths, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SimpleCNN(num_classes=len(set(Y)))
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.cuda(), labels.cuda()

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.cuda(), labels.cuda()
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
            f"Validation Loss: {val_loss/len(val_loader):.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            best_fold = fold + 1
            print(f"New best model found! Validation Accuracy: {best_val_accuracy:.2f}%")

    print(f"Fold {fold + 1} / {skf.n_splits}")

torch.save(best_model, f'best_model_fold{best_fold}.pth')
print(f"Best model saved from fold {best_fold} with accuracy: {best_val_accuracy:.2f}%")