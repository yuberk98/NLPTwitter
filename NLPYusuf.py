import tweepy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Twitter API anahtarlarını girin
consumer_key = 'hxCwsEreqBryQL3KGHLsdMRDe'
consumer_secret = 'PG6H8SEVkAYELRITZErVGQf3QO34XWq03aCivHXma3vPa5ndqR'
access_token = '2288163702-IIdLuynrx9BtmSdVgVoPdUzjC9pBv3j1DVZLN8s'
access_token_secret = '6bXWSCU94hvrGVapjtNYqA1O340pgAwSlldid6W22GdVW'

# API ile iletişim kurma
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Twitter'dan Bitcoin ile ilgili tweet'leri toplama
tweets = tweepy.Cursor(api.search_tweets, q='bitcoin', lang='en').items(50)

# Tweet'lerdeki duyguları analiz etme
data = []
for tweet in tweets:
    analysis = TextBlob(tweet.text)
    sentiment = analysis.sentiment.polarity
    data.append(sentiment)

# Veriyi pandas DataFrame'e dönüştürme
data_frame = pd.DataFrame(data, columns=['sentiment'])

# Veriyi görselleştirme
plt.figure(figsize=(10, 6))
data_frame['sentiment'].plot(kind='hist', bins=50)
plt.title('Bitcoin ile ilgili Tweet Duyguları')
plt.xlabel('Duygu')
plt.ylabel('Frekans')
plt.show()
