import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

#plt.ion()
df = pd.read_csv('customer_transaction.csv')
df.info()
df_features = df[['is_Alerted', 'is_Suspicious', 'transaction_amount', 'correspondent_bank',
                  'debit_credit', 'Account_type', 'Account_Classification', 'Risk_level', 
                  'Annual_income', 'is_noncitizen']]
df_features.info()
df_transformed = pd.get_dummies(df_features)
df_transformed.info()
#exploring the correlation between variables in the df_transformed  
df_transformed.corr().head(2).transpose()
sns.heatmap(df_transformed.corr())
plt.show()
sns.jointplot(x='transaction_amount',y='Annual_income',data=df_features,kind='reg')
plt.show()
sns.pairplot(df_features,hue='is_Suspicious')
plt.show()
#The train_test_split function is used to split the data into training and testing sets
X = df_transformed.drop(['is_Alerted', 'is_Suspicious'], axis=1)
y = df_transformed['is_Suspicious']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Using LogisticRegression
logmodel = LogisticRegression()
#The model is trained on the training data using logmodel.fit(X_train, y_train)
logmodel.fit(X_train,y_train)
#Predictions are made on the test data using logmodel.predict(X_test)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
cf = confusion_matrix(y_test,predictions)
TP = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TN = cf[1][1]
recall = TP/(TP+FN)
accuracy = TP/(TP+FP)
print("Accuracy on testing data: {:.4f} \n\nRecall on testing data: {:.4f}".format(accuracy,recall))

#Using Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
cf = confusion_matrix(y_test,predictions)
cf
TP = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TN = cf[1][1]
recall = TP/(TP+FN)
accuracy = TP/(TP+FP)
print("Accuracy on testing data: {:.4f} \n\nRecall on testing data: {:.4f}".format(accuracy,recall))

# Using Support Vector Machine Classifier
svc = SVC()
svc.fit(X_train, y_train)
predictions = svc.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
cf = confusion_matrix(y_test, predictions)
cf
TP = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TN = cf[1][1]
recall = TP / (TP + FN)
accuracy = TP / (TP + FP)
print("Accuracy on testing data: {:.4f} \n\nRecall on testing data: {:.4f}".format(accuracy, recall))

# Using Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
predictions = gbc.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
cf = confusion_matrix(y_test, predictions)
cf
TP = cf[0][0]
FP = cf[0][1]
FN = cf[1][0]
TN = cf[1][1]
recall = TP / (TP + FN)
accuracy = TP / (TP + FP)
print("Accuracy on testing data: {:.4f} \n\nRecall on testing data: {:.4f}".format(accuracy, recall))
