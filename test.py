def load_dataset(audio_dir):
    features = []
    y = []

    for idx, filename in enumerate(os.listdir(audio_dir)):
        if filename.endswith('.wav'):
            # Convert to WAV first
            # wav_path = os.path.join(audio_dir, filename.replace('.m4a', '.wav'))
            # convert_m4a_to_wav(os.path.join(audio_dir, filename), wav_path)

            # Extract features
            file_path = os.path.join(audio_dir, filename)
            mfcc_features = extract_mfcc_features(file_path)
            # print(mfcc_features.shape)
            features.append(mfcc_features)

            # Append corresponding label (for example, assuming labels is a dict: {filename: label})
            y.append(filename)

    return np.array(features)