def run_trained_model(X):
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.transform(X)
    
    import gdown
    url = 'https://drive.google.com/file/d/1eGcwmP4SxKkUKKXGlEEv6Y-8rilwtfgw/view?usp=drive_link'
    output = "new_model.pkl"
    gdown.download(url, output, fuzzy=True)
    
    import pickle
    with open(output,'rb') as f:
        model = pickle.load(f)
    
    
    predictions = model(X_scaled)
    assert predictions.shape == Y.shape
    return predictions
