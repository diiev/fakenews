import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models.model import train_model, predict
from utils.preprocessing import preprocess_data

# Загрузка данных
data = pd.read_csv('data/news_data.csv')

# Предобработка данных
x_train, x_test, y_train, y_test = preprocess_data(data)

# Обучение модели
model, vectorizer = train_model(x_train, y_train)

# Предсказание
y_pred = predict(model, vectorizer, x_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Визуализация матрицы ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
plt.xlabel('Предсказания')
plt.ylabel('Истинные метки')
plt.title('Матрица ошибок')
plt.show()

# Визуализация распределения предсказаний
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette='viridis')
plt.title('Распределение предсказаний')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.show()