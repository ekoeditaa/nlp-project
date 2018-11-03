import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

# Define your dataset path here
DATA_PATH = 'CellPhoneReview.json'
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
a = 0.2

def printMenu():
	print("""
####################################################
        Binary Sentiment Classification Apps
####################################################
	
Option:
1. Set Ratio Training-Testing Dataset
2. Train, Test, and Get Accuracy of the Model
3. Print Top-N words representing Positive sentiment
4. Print Top-N words representing Negative sentiment
5. Your own sentence Sentiment Classification
6. Exit

####################################################
""")

printMenu()

while 1:
	                  
	userChoice=int(input("\nType in your choice: "))
	if userChoice == 1:
		a = float(input("Test size (0 - 1): "))	
		while a > 1 or a < 0:
			a = float(input("Please select a valid Test size (0 - 1): "))	
	elif userChoice == 2:
		#Train the model
		print("Test size: ", a)
		#Split the dataset to training and testing
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=a)
		print('# train records: {}'.format(X_train.shape[0]))
		print('# test records: {}'.format(X_test.shape[0]))
		X_train_c = cv.fit_transform(X_train)
		X_test_c = cv.transform(X_test)
		model = lr.fit(X_train_c, Y_train)
		y_pred = lr.predict(X_test_c)
		acc = metrics.accuracy_score(Y_test, y_pred)
		print('Model accuracy: {}'.format(acc))


	elif userChoice == 3:
		try:
			n = int(input("N: "))
			word = cv.get_feature_names()
			coef = model.coef_.tolist()[0]
			coeff_df = pd.DataFrame({'Word': word, 'Coefficient': coef})
			coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0,1])
			print('\n######### Top {} Positive ###########'.format(n))
			print(coeff_df.head(n).to_string(index=False))	
		except:
			print("Please train a model first!")
			continue
	elif userChoice == 4:
		try:
			n = int(input("N: "))
			word = cv.get_feature_names()
			coef = model.coef_.tolist()[0]
			coeff_df = pd.DataFrame({'Word': word, 'Coefficient': coef})
			coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0,1])
			print('\n######### Top {} Positive ###########'.format(n))
			print(coeff_df.tail(n).to_string(index=False))	
		except:
			print("Please train a model first!")
			continue
	elif userChoice == 5:
		try:
			userInput = [input("Please type in your sentence: ")]
			if lr.predict(cv.transform(userInput)) == [0]:
				print("Your sentence sentiment classification is NEGATIVE")
			else:
				print("Your sentence sentiment classification is POSITIVE")			
		except:
			print("Please train a model first!")
			continue
	elif userChoice == 6:
		break
	else:
		printMenu()


