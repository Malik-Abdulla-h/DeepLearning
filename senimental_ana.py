from textblob import TextBlob
from newspaper import Article
import nltk

# nltk.download('punkt')
# nltk.download('punkt_tab')

url = "https://en.wikipedia.org/wiki/Mathematics"
article = Article(url)

article.download()
article.parse()
article.nlp()

text = article.summary

blob = TextBlob(text)
sentiment = blob.sentiment.polarity

print(f"Sentiment Polarity: {sentiment}")



