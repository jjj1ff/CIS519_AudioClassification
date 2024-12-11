def run_trained_model(X):
  # X is of shape (N, ), where each element is a path string to a WAV file.
  X_list = []
  def get_mfcc_features(file_path):
    data, sr = librosa.load(file_path)
    
    data = np.array(2*((data-np.min(data))/(np.max(data)-np.min(data)))-1)
    data = librosa.effects.time_stretch(data,rate=1.3)
    data = librosa.effects.pitch_shift(data,sr=sr,n_steps=2)
    data_mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    audio_data = np.mean(data_mfcc, axis=1)
    return audio_data
  for x in X:
    if x.endswith(".wav"):
      mfcc_features = get_mfcc_features(x)
      X_list.append(mfcc_features)

  X_list = np.array(X_list)
  import gdown
  url = 'https://drive.google.com/file/d/1xP54Qp_0t33X6le1Ffis9nYXcBukiHYB/view?usp=sharing'

  output = "best_model.pkl"
  gdown.download(url, output, fuzzy=True)


  import pickle
  import sklearn
  with open(output,'rb') as f:
    model = pickle.load(f)


  predictions = model.predict(X_list)
  assert predictions.shape == Y.shape
  return predictions
