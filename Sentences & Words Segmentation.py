import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import json
import os

###################################### Specify the path to the json text
file_folder = "D:\\Isolated Box\\"
file_name   = "SampleReview.json"
#file_name   = "CellPhoneReview.json"
file_path   = os.path.join(file_folder, file_name)
print("File Path : {}\n".format(file_path))

###################################### Read and parse the json text
#### Parse the json text to raw_json
raw_json    = []
with open(file_path, mode='r') as file_json:
    for line in file_json:
        raw_json.append(json.loads(line))

#### Get the json tag for reviewText and asin
tag_json    = list(raw_json[0].keys())
key_product = tag_json[1]
key_review  = tag_json[2]

###################################### Functions to tokenize words and sentences as well as stemming
#### Tokenize text to words then add to the json
def token_words(json_data, tag_text, tag_token='token', tag_words="totWords"):
    max, min, first = 0, 0, True
    for data in json_data:
        words = nltk.word_tokenize(data[tag_text])
        count = len(words)
        data[tag_token] = words
        data[tag_words] = count
        if max < count:
            max   = count
        if first:
            min   = count
            first = False
        if min > count:
            min   = count
    return json_data, min, max

#### Tokenize text to sentences then add to the json
def token_sents(json_data, tag_text, tag_token='token', tag_sents="totSents"):
    max, min, first = 0, 0, True
    for data in json_data:
        sents = nltk.sent_tokenize(data[tag_text])
        count = len(sents)
        data[tag_token] = sents
        data[tag_sents] = count
        if max < count:
            max   = count
        if first:
            min   = count
            first = False
        if min > count:
            min   = count
    return json_data, min, max

#### Stem text then add to the json
def stem_text(json_data, tag_text, tag_stemmed="stemmedText", tag_stems="totStems", stem_type="Snowball"):
    stemmer = nltk.SnowballStemmer('english') if stem_type == "Snowball" else \
              nltk.PorterStemmer() if stem_type == "Porter" else \
              nltk.LancasterStemmer()
    max, min, first = 0, 0, True
    for data in json_data:
        stems = [stemmer.stem(word) for word in nltk.word_tokenize(data[tag_text])]
        count = len(stems)
        data[tag_stemmed] = stems
        data[tag_stems]   = count
        if max < count:
            max   = count
        if first:
            min   = count
            first = False
        if min > count:
            min   = count
    return json_data, min, max

#### Modify the json to include words and sentences token and stems
key_rev_words      = "totReviewWords"
key_token_word     = "tokenWords"
key_rev_sents      = "totReviewSents"
key_token_sent     = "tokenSents"
key_stemmed_review = "stemmedReviewText"
key_rev_stems      = "totStemmedReviewWords"
raw_json, min_words, max_words = token_words(raw_json, key_review, key_token_word    , key_rev_words)
raw_json, min_sents, max_sents = token_sents(raw_json, key_review, key_token_sent    , key_rev_sents)
raw_json, min_stems, max_stems = stem_text  (raw_json, key_review, key_stemmed_review, key_rev_stems)
print("""WORDS
Min : {}
Max : {}
SENTS
Min : {}
Max : {}
STEMS
Min : {}
Max : {}
""".format(min_words, max_words, min_sents, max_sents, min_stems, max_stems))

###################################### Plot the words, sentences, stems distribution
array_cnt_words = np.zeros((max_words + 1), dtype='int32')
array_cnt_sents = np.zeros((max_sents + 1), dtype='int32')
array_cnt_stems = np.zeros((max_stems + 1), dtype='int32')

for data in raw_json:
    array_cnt_words[data[key_rev_words]] += 1
    array_cnt_sents[data[key_rev_sents]] += 1
    array_cnt_stems[data[key_rev_stems]] += 1

plt.plot(array_cnt_sents, 'b-')
plt.title("Sentences Distribution")
plt.legend(['Sentences'])
plt.show()
plt.plot(array_cnt_words, 'r-', array_cnt_stems, 'g-')
plt.title("Words and Stems Distribution")
plt.legend(['Words', 'Stems'])
plt.show()

###################################### Segment 5 sentences
#### Sort the json from low total sentences to high total sentences
raw_json_sorted  = sorted(raw_json, key=lambda k: k[key_rev_sents])

#### Choose 5 sentences, low and high total sentences
sentences        = []
index_first      = 0
index_last       = len(raw_json_sorted) - 1
even_i           = True
tot_sentences    = 5
for i in range(tot_sentences):
    if even_i:
        sentences.append(raw_json_sorted[index_first][key_review])
        index_first += 1
        even_i = False
    else:
        sentences.append(raw_json_sorted[index_last][key_review])
        index_last  -= 1
        even_i = True

#### Segment the 5 sentences
sentences_sgmnt = [nltk.sent_tokenize(sent) for sent in sentences]

for i in range(tot_sentences):
    print('#####')
    print('Sentence {}:\n{}\n'.format(i+1, sentences[i]))
    print('Segmented:')
    j = 1
    for sent in sentences_sgmnt[i]:
        print(j, ":\n", sent)
        j += 1
    print('#####')
print(raw_json_sorted==raw_json)

###################################### Extract top 20 words and stems
#### Define stopwords
stopwords   = nltk.corpus.stopwords.words('english') + list(string.punctuation)

#### Function to get words frequency distribution
def words_dist(json_data, tag_token):
    return nltk.FreqDist(
        word.lower()                        \
        for data in json_data               \
        for word in data[tag_token]         \
        if word.lower() not in stopwords    \
        )

#### Get the stems and words distribution
stems_dist = words_dist(raw_json, key_stemmed_review)
words_dist = words_dist(raw_json, key_token_word    )

#### Get the top 20 words and stems
top_words  = 20
print("\n##### WORDS #####")
print('Top {} words:'.format(top_words))
for word in words_dist.most_common(top_words):
    print(word, end=', ')
print("\n##### ##### #####")
print("\n##### STEMS #####")
print('Top {} stems:'.format(top_words))
for stem in stems_dist.most_common(top_words):
    print(stem, end=', ')
print("\n##### ##### #####")
print('\n##### STOPWORDS #####')
print(stopwords)
