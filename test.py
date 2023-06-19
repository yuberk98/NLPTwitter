import sys
import time
import numpy as np
import nltk
import requests
from bs4 import BeautifulSoup
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel, QPushButton
from PyQt5.QtCore import QTimer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
from pycoingecko import CoinGeckoAPI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from deep_learning_model import create_deep_learning_model

nltk.download('vader_lexicon')  # VADER lexicon'u indir

edge_options = Options()
edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
edge_options.add_argument("--disable-extensions")
edge_options.add_argument("--disable-gpu")
edge_options.add_argument("--no-sandbox")
edge_options.add_argument("--disable-dev-shm-usage")
edge_options.add_argument("--remote-debugging-port=9222")
edge_options.add_argument("--user-data-dir=./edge-user-data")
edge_options.add_argument('--log-level=3')  # 3: ERROR, 2: WARNING, 1: INFO, 0: DEBUG

service = Service("msedgedriver.exe")
driver = webdriver.Edge(service=service, options=edge_options)  # EDGEdriver yolunu düzenleyin

def manual_login_to_twitter(driver):
    login_url = "https://twitter.com/login"
    driver.get(login_url)
    time.sleep(3)

    print("Lütfen tarayıcıda Twitter'a giriş yapın ve devam etmek için Enter tuşuna basın.")
    input()

def get_tweets_scraping(driver, search_term="bitcoin", count=100):
    url = f"https://twitter.com/search?q={search_term}&src=typed_query"
    driver.get(url)
    time.sleep(3)

    tweet_texts = []
    for _ in range(count // 20):
        tweets = driver.find_elements(By.CSS_SELECTOR, "article div[lang='en']")
        for tweet in tweets:
            tweet_texts.append(tweet.text)
            if len(tweet_texts) >= count:
                break
        if len(tweet_texts) < count:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)

    return tweet_texts[:count]

def analyze_sentiment(tweets):
    if len(tweets) == 0:
        return 0

    sentiment = 0
    sia = SentimentIntensityAnalyzer()
    for tweet in tweets:
        sentiment_score = sia.polarity_scores(tweet)
        sentiment += sentiment_score['compound']
    return sentiment / len(tweets)

def get_bitcoin_price():
    cg = CoinGeckoAPI()
    bitcoin_data = cg.get_price(ids='bitcoin', vs_currencies='usd')
    return bitcoin_data['bitcoin']['usd']

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Bitcoin Duygu Analizi')
        self.setGeometry(300, 300, 600, 400)

        layout = QVBoxLayout()

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.btc_label = QLabel(self)
        layout.addWidget(self.btc_label)

        self.sentiment_label = QLabel(self)
        layout.addWidget(self.sentiment_label)

        self.start_button = QPushButton("Başlat", self)
        self.start_button.clicked.connect(self.start_analysis)
        layout.addWidget(self.start_button)

        self.setLayout(layout)

    def start_analysis(self):
        self.start_button.setEnabled(False)
        self.start_button.setText("Analiz Başladı")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_info)
        self.timer.start(60000)  # 1 dakika
        self.update_info()

    def update_info(self):
        # Burada Twitter'dan tweetleri alın, duygu analizini yapın ve Bitcoin fiyatını elde edin.
        # Örnek olarak şu anlık değerler kullanıyoruz:
        tweets = ["Örnek Tweet 1", "Örnek Tweet 2", "Örnek Tweet 3"]
        sentiment = 0.1
        bitcoin_price = 50000

        self.log_text.clear()
        for i, tweet in enumerate(tweets):
            self.log_text.append(f"{i + 1}. {tweet}")

        self.btc_label.setText(f"Anlık Bitcoin Fiyatı: ${bitcoin_price}")
        self.sentiment_label.setText(f"Twitter'daki Duygu Analizi Sonucu: {sentiment}")



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


# Derin öğrenme modelini oluşturalım ve eğitelim
model = create_deep_learning_model(input_dim=1)
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=0)

# Veriyi eğitim ve test veri setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik Regresyon modelini eğitelim
model = LogisticRegression()
model.fit(X_train, y_train)

# Modelin doğruluğunu test veri seti üzerinde ölçelim
y_pred = (model.predict(X_test) > 0.5).astype('int32')
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
