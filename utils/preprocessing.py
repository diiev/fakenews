from sklearn.model_selection import train_test_split

def preprocess_data(data):
    x = data['text']
    y = data['label']
    return train_test_split(x, y, test_size=0.2, random_state=42)