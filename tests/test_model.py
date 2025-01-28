import unittest
import pandas as pd
from models.model import train_model, predict
from utils.preprocessing import preprocess_data

class TestModel(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('data/news_data.csv')
        self.x_train, self.x_test, self.y_train, self.y_test = preprocess_data(self.data)
        self.model, self.vectorizer = train_model(self.x_train, self.y_train)

    def test_predict(self):
        y_pred = predict(self.model, self.vectorizer, self.x_test)
        self.assertEqual(len(y_pred), len(self.y_test))

if __name__ == '__main__':
    unittest.main()