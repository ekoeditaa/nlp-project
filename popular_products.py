import pandas as pd
import json

# Define your dataset path here
DATA_PATH = './data/reviews_Cell_Phones_and_Accessories_5.json'

# Load the dataset as dataframe
df = pd.read_json(DATA_PATH, orient="columns", lines=True)

# Get the top 10 products with most reviews
print(df['asin'].value_counts()[:10])

# Get the top 10 reviewers who give most reviews
print(df['reviewerID'].value_counts()[:10])