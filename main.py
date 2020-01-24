# IMPORTS
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# READ COMMENTS
file_data = pd.read_csv("irfan_dataset.csv")
test_data = pd.read_csv("comment_file.csv")

# CREATE VECTORIZER
count_vectorizer = CountVectorizer(analyzer='word',ngram_range=(1, 1),stop_words=None, token_pattern = '[a-zA-Z]+')
transformed_vector = count_vectorizer.fit_transform(file_data["COMMENTS"].values.astype('U'))
dataframe = pd.DataFrame(transformed_vector.toarray(),columns=count_vectorizer.get_feature_names())

# SELECT KBEST
selector = SelectKBest(score_func=chi2, k=1000)
selector.fit(dataframe, file_data["PREDICTION"])
extracted_data = selector.transform(dataframe)
extracted_feature = np.asarray(count_vectorizer.get_feature_names())[selector.get_support()]
dataframe = dataframe[extracted_feature]

# INITIALIZATION
tokens_list = dataframe.columns.values

# SPLIT TEST TRAIN
data = dataframe[tokens_list[0:len(tokens_list)]]
label = file_data["PREDICTION"]
train_features, test_features, train_labels, test_labels = train_test_split(data, label, test_size = 0.3,shuffle=False)

# CREATE MODEL
model = RandomForestClassifier()

# TRAIN MODEL
trainedModel = model.fit(train_features, train_labels)

# PREDICTION
predictions = trainedModel.predict(test_features)

# SHOW PREDICTIONS
print(sklearn.metrics.accuracy_score(test_labels, predictions))
print('F1 Score - ', {f1_score(test_labels, predictions,average='macro')})
print(precision_score(test_labels, predictions, average='macro'))
print(recall_score(test_labels, predictions, average='macro'))

# SHOW ACCURACY
print(accuracy_score(test_labels, predictions))

# PREPROCESS INPUT DATA
#input = "i love you you are my life"
#input = "fuck off you fucking nigger"
#input=[input]
transformed_vector = count_vectorizer.transform(test_data["COMMENTS"].values.astype('U'))
transformed_dataframe = pd.DataFrame(transformed_vector.toarray(), columns=count_vectorizer.get_feature_names())
transformed_dataframe = transformed_dataframe[extracted_feature]

# TESTING
print(dataframe)
#dataframe.to_csv("./test_dataframe_1", encoding='utf-8', index=False)
#transformed_dataframe.to_csv("./test_dataframe_2", encoding='utf-8', index=False)

# PREDICTION
predictions = model.predict(transformed_dataframe)
predictions[predictions == 0] = 1
print("FINAL ACCURACY")
print(accuracy_score(test_data["PREDICTION"], predictions))
np.savetxt("predictions.csv", predictions, delimiter=",",fmt="%s")

print(predictions)