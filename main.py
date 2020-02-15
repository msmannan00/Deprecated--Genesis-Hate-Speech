# IMPORTS
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# READ COMMENTS
train_data = pd.read_csv("training_data.csv")
test_data = pd.read_csv("input_data.csv")

# CLEANING
train_data['COMMENTS'] = train_data['COMMENTS'].map(lambda x: re.sub(r'[^A-Za-z ]+', '', x))
test_data['COMMENTS'] = test_data['COMMENTS'].map(lambda x: re.sub(r'[^A-Za-z ]+', '', x))

# CREATE VECTORIZER
count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=None, token_pattern='[a-zA-Z]+')
transformed_vector = count_vectorizer.fit_transform(train_data["COMMENTS"].values.astype('U'))
dataframe = pd.DataFrame(transformed_vector.toarray(), columns=count_vectorizer.get_feature_names())

# SELECT KBEST
selector = SelectKBest(score_func=chi2, k=3000)
selector.fit(dataframe, train_data["PREDICTION"])
extracted_data = selector.transform(dataframe)
extracted_feature = np.asarray(count_vectorizer.get_feature_names())[selector.get_support()]
dataframe = dataframe[extracted_feature]

# INITIALIZATION
tokens_list = dataframe.columns.values

# SPLIT TEST TRAIN
data = dataframe[tokens_list[0:len(tokens_list)]]
label = train_data["PREDICTION"]
train_features, test_features, train_labels, test_labels = train_test_split(data, label, test_size=0.3, shuffle=False)

# CREATE MODEL
model = RandomForestClassifier(max_depth=150, random_state=10)

# TRAIN MODEL
trainedModel = model.fit(train_features, train_labels)

# PREDICTION
predictions = trainedModel.predict(test_features)

# SHOW PREDICTIONS
print(sklearn.metrics.accuracy_score(test_labels, predictions))
print('F1 Score - ', {f1_score(test_labels, predictions, average='macro')})
print(precision_score(test_labels, predictions, average='macro'))
print(recall_score(test_labels, predictions, average='macro'))

# SHOW ACCURACY
print(accuracy_score(test_labels, predictions))

# PREPROCESS OF INPUT
transformed_vector = count_vectorizer.transform(test_data["COMMENTS"].values.astype('U'))
transformed_dataframe = pd.DataFrame(transformed_vector.toarray(), columns=count_vectorizer.get_feature_names())
transformed_dataframe = transformed_dataframe[extracted_feature]

# TESTING OF INPUT
transformed_dataframe.to_csv("./feature_matrix", encoding='utf-8', index=False)

# PREDICTION OF INPUT
predictions = model.predict(transformed_dataframe)

# ACCURACY OF INPUT
print(accuracy_score(test_data["PREDICTION"], predictions))
np.savetxt("predictions.csv", predictions, delimiter=",", fmt="%s")
