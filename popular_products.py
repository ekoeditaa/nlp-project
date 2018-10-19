import pandas as pd
import json

DATA_PATH = './data/reviews_Cell_Phones_and_Accessories_5.json'

df = pd.read_json(DATA_PATH, orient="columns", lines=True)
print(df.head(10))