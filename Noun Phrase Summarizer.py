import numpy as np
import pandas as pd
import nltk
import os
import random
from np_detector import np_detector

###################################### Specify the path to the json text
file_folder = "D:\\Isolated Box\\"
file_name   = "SampleReview.json"
#file_name   = "CellPhoneReview.json"
file_path   = os.path.join(file_folder, file_name)
print("File Path : {}\n".format(file_path))

###################################### Read the json text
#### Get the json as pandas table
df = pd.read_json(file_path, orient="columns", lines=True)

#### Take the review texts only
df_review_text = df['reviewText']

###################################### Detect NPs from the review texts
print("\n######################################     Detect NPs from the review texts    ######################################")
#### Detect the NPs
np_list = np_detector(df_review_text).np_list

#### Extract top 20 NPs
fd  = nltk.FreqDist(np_list)
top = 20
print("Top 20 NPs:\n{}".format(fd.most_common(top)))

######################################## Detect 10 representative NPs from 3 popular products
print("\n############################### Detect 10 representative NPs from 3 popular products ################################")
#### Get top 3 products
top_3_prod = df['asin'].value_counts()[:3]
top_3 = []
for prod, count in top_3_prod.iteritems():
    top_3.append(prod)
print('Top 3 products:\n{}\n'.format(top_3_prod))

#### Detect 10 representative NPs for each product
top = 10
for prod in top_3:
    df2 = df.loc[df['asin'] == prod]
    df2_review_text = df2['reviewText']
    sentences_nps   = np_detector(df2_review_text, pandas=True).np_list
    fd  = nltk.FreqDist(sentences_nps)
    print('#####')
    print('{} representative NP:'.format(prod))
    print(fd.most_common(top))
    print('#####')

###################################### Detect NPs from randomly 5 review texts
print("\n###################################### Detect NPs from randomly 5 review texts ######################################")
tot_length      = len(df_review_text)
tot_sentences   = 5
sentences       = []

for i in range(tot_sentences):
    index           = random.randint(0, tot_length-1)
    df2             = df.loc[index]
    df2_review_text = df2['reviewText']
    sentences_nps   = np_detector([df2_review_text], pandas=False).np_list
    print('#####')
    print('Sentence {}:\n{}'.format(i+1, df2_review_text))
    print('NPs:\n{}'        .format(sentences_nps))
    print('#####')
