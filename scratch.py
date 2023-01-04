#import nltk
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context
#nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
#print(sia.polarity_scores('I am very happy'))

def get_sentiment(text:str):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    sentiment = 'Positive' if score.get('compound') >=0.5 else \
        'Negative' if score.get('compound') <=0.5 else 'Neutral'
    return sentiment

if __name__ == '__main__':
    print(get_sentiment('I am very'))
