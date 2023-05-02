import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Örnek etiketli veri (duygu analizi değeri ve Bitcoin fiyatındaki değişime göre etiketler)
data = np.array([
    [0.1, 1],
    [-0.2, 0],
    [0.3, 1],
    [-0.4, 0],
    [0.2, 1],
    [-0.1, 0],
])

X = data[:, 0].reshape(-1, 1)  # Duygu analizi değerleri
y = data[:, 1]  # Etiketler (0: düşüş, 1: artış)

# Veriyi eğitim ve test veri setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik Regresyon modelini eğitelim
model = LogisticRegression()
model.fit(X_train, y_train)

# Modelin doğruluğunu test veri seti üzerinde ölçelim
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy}")


