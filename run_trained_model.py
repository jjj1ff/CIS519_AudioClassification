def run_trained_model(X):
  # X is of shape (N, ), where each element is a path string to a WAV file.
  # TODO: featurize the WAV files

  # TODO: load your model weights
  def download_model_weights():
    import gdown
    url = 'https://drive.google.com/file/d/1BRlqdsi5WGemSe1jClI5rfuRuL96CUB7/view?usp=drive_link'
    output = "my_weights.npy"
    gdown.download(url, output, fuzzy=True)
    return output
  weight_path = download_model_weights()
  weights = np.load(weight_path, allow_pickle=True)

  # TODO: setup model
  def random_classifier(X):
    return np.random.randint(0, 7, X.shape[0])

  predictions = random_classifier(X) # Should be shape (N,) where each element is a class integer for the corresponding data point.
  assert predictions.shape == Y.shape
  return predictions