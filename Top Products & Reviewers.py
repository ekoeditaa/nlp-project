import nltk
import json
import os

###################################### Specify the path to the json text
file_folder = "D:\\Isolated Box\\"
file_name   = "SampleReview.json"
#file_name   = "CellPhoneReview.json"
file_path   = os.path.join(file_folder, file_name)
print("File Path :", file_path)

###################################### Read and parse the json text
#### Parse the json text to raw_json
raw_json = []
with open(file_path, mode='r') as file_json:
    for line in file_json:
        raw_json.append(json.loads(line))

#### Get the json tag for reviewerID and asin
tag_json    = list(raw_json[0].keys())
key_product = tag_json[1]
key_user    = tag_json[0]
print("Tags :\n"     , tag_json)
print("Tag Product :", key_product)
print("Tag User :"   , key_user)

###################################### Find the frequency distribution
dist_product = nltk.FreqDist(data[key_product] for data in raw_json)
dist_user    = nltk.FreqDist(data[key_user   ] for data in raw_json)

#### Find top 10 product and user
top = 10
print("Top {} Product:\n{}".format(top, dist_product.most_common(top)))
print("Top {} User   :\n{}".format(top, dist_user.most_common(top)))
dist_product.plot(title="Product Distribution")
dist_user.plot(title="User Distribution")

#### List the total product and user
dist_prod_user = nltk.ConditionalFreqDist((data[key_product], data[key_user])
                                          for data in raw_json)
print("\nTabulate")
dist_prod_user.tabulate()
#dist_prod_user.plot()
