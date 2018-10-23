import nltk
import json
import os
import random

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

#### Get the json tag for reviewText
tag_json    = list(raw_json[0].keys())
key_review  = tag_json[2]

###################################### Randomly get 5 sentences
raw_json_length = len(raw_json)
tot_sentences   = 5
sentences       = []
for i in range(tot_sentences):
    index = random.randint(0, tot_sentences-1)
    sentences.append(raw_json[index][key_review])
sentences_pos   = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

###################################### POS Tagging
#### Print Penn Treebank POS reference
print('##### Penn Treebank POS Tag #####')
nltk.help.upenn_tagset()
print('#################################\n')

#### POS tag the 5 sentences
for i in range(tot_sentences):
    print('#####')
    print('Sentence {}:\n{}'.format(i+1, sentences[i]))
    print('POS Tagging:\n{}'.format(sentences_pos[i]))
    print('#####')
    
