from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

def train_model(x_train, y_train):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    
    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(x_train_tfidf, y_train)
    
    return model, vectorizer

def predict(model, vectorizer, x_test):
    x_test_tfidf = vectorizer.transform(x_test)
    return model.predict(x_test_tfidf)