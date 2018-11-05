import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Define your dataset path here
DATA_PATH = './data/reviews_Cell_Phones_and_Accessories_5.json'

# Load the dataset as dataframe
df = pd.read_json(DATA_PATH, orient="columns", lines=True)

# Get rid of score 3 revies
df_rating = df.loc[(df.overall != 3)]
X = df_rating['reviewText']
#Use binary numbers 0 to present negative and 1 to present positive reviews
y_dict = {1:0, 2:0, 4:1, 5:1}
Y = df_rating['overall'].map(y_dict)

#Convert text to a matrix of token counts
cv = TfidfVectorizer(stop_words = 'english', lowercase = True)

#Use logistic regression model on word count
lr = LogisticRegression()

#Train the model
#Split the dataset to training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print('# train records: {}'.format(X_train.shape[0]))
print('# test records: {}'.format(X_test.shape[0]))

X_train_c = cv.fit_transform(X_train)
X_test_c = cv.transform(X_test)
model = lr.fit(X_train_c, Y_train)
y_pred = lr.predict(X_test_c)
acc = metrics.accuracy_score(Y_test, y_pred)
print('Model accuracy: {}'.format(acc))

print('Confusion Matrix')
print(metrics.confusion_matrix(Y_test, y_pred))

#Find the top 20 positive and negative words
word = cv.get_feature_names()
coef = model.coef_.tolist()[0]
coeff_df = pd.DataFrame({'Word': word, 'Coefficient': coef})
coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0,1])
print('\n######### Top 20 Positive ###########')
print(coeff_df.head(20).to_string(index=False))
print('\n######### Top 20 Negative ###########')
print(coeff_df.tail(20).to_string(index=False))