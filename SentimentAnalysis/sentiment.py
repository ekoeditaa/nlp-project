import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Define your dataset path here
DATA_PATH = './data/reviews_Cell_Phones_and_Accessories_5.json'

# Load the dataset as dataframe
df = pd.read_json(DATA_PATH, orient="columns", lines=True)

# Take the review texts only
df_rating = pd.DataFrame({'rating': df.overall, 'review': df.reviewText})

#Splitting positive and negative datasets based on rating
df_positive = df_rating.loc[(df_rating.rating >= 3)]
df_negative = df_rating.loc[(df_rating.rating < 3)]

default_stopwords = set(stopwords.words('english'))

#Using TweetTokenizer for apostrophe problem
tokenizer = TweetTokenizer()

for df in [df_positive, df_negative]:
	words = []
	for idx, text in df.iterrows():
	  temp = tokenizer.tokenize(text['review'])
	  temp = [word.lower() for word in temp if (len(word) > 1 and not word.isnumeric() and word not in default_stopwords)]
	  for word in temp:
	  	words.append(word)

	fdist = nltk.FreqDist(words)

	print("#########################")

	for word, frequency in fdist.most_common(20):
		print(u'{};{}'.format(word,frequency))

#(df_rating.loc[(df_rating.rating >= 3)])