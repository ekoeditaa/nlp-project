import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

# Define your dataset path here
DATA_PATH = './data/reviews_Cell_Phones_and_Accessories_5.json'

# Load the dataset as dataframe
df = pd.read_json(DATA_PATH, orient="columns", lines=True)

# Take the review texts only
df_review_text = df['reviewText']

# Iterate to find no of sentences of each review
lengths = []
for idx, text in df_review_text.iteritems():
  lengths.append(len(sent_tokenize(text)))

# Add the sentences length into the data frame
sentence_lengths = pd.Series(lengths)
df['Review Text Lengths'] = sentence_lengths.values

# Get the no of reviews with certain lengths
reviews_length_series = df['Review Text Lengths'].value_counts()
eviews_length_series.plot(kind="bar", use_index=False) # try line also. See which one we want
