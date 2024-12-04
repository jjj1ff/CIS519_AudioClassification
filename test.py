import os
import librosa
import numpy as np
# !pip install ffmpeg
import ffmpeg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def extract_mfcc_features(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  # Extract MFCC features
    mfccs_mean = np.mean(mfccs, axis=1)  # Take the mean of MFCCs over time
    return mfccs_mean

def load_dataset(audio_dir):
    features = []
    y = []

    for idx, filename in enumerate(os.listdir(audio_dir)):
        if filename.endswith('.wav'):

            file_path = os.path.join(audio_dir, filename)
            mfcc_features = extract_mfcc_features(file_path)

            features.append(mfcc_features)

            y.append(filename)

    return np.array(features)

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

base_dir = "DATA"
X = []
Y = []

for idx, subfolder in enumerate(os.listdir(base_dir)):
    subfolder_path = os.path.join(base_dir, subfolder)


    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")



        x = load_dataset(subfolder_path)
        y = np.array([CLASS_TO_LABEL[subfolder]] * x.shape[0])
        print(x.shape, y.shape)

        X.append(x)
        Y.append(y)
X = np.concatenate(X, 0)
Y = np.concatenate(Y, 0)
print(X.shape)
print(Y.shape)
print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
for i in range(len(y_pred)):
  print(f'Correct {(y_pred[i] == y_test[i])}, Pred {y_pred[i]}, Label: {y_test[i]}')

base_dir = "EVAL"
eval_X = []
eval_Y = []

for idx, subfolder in enumerate(os.listdir(base_dir)):
    subfolder_path = os.path.join(base_dir, subfolder)


    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_path}")

        x = load_dataset(subfolder_path)
        y = np.array([CLASS_TO_LABEL[subfolder]] * x.shape[0])
        if x.shape[0] == 0:
          continue

        print(x.shape, y.shape)
        eval_X.append(x)
        eval_Y.append(y)

eval_X = np.concatenate(eval_X, 0)
eval_Y = np.concatenate(eval_Y, 0)

eval_X_scaled = scaler.transform(eval_X)

y_pred = model.predict(eval_X_scaled)
accuracy = accuracy_score(eval_Y, y_pred)
print(f"Model Accuracy on unseen data: {accuracy:.2f}")