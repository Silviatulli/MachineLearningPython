#Naive Bayes Classifier
from pydataset import data
iris=data('iris')

x=iris.drop('Species',axis=1)
y=iris['Species']

#splitting the dataset
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2,random_state=0)


# Model creation
# for CONTINUES INDEPENDENT VARIABLES
from sklearn.naive_bayes import GaussianNB
model=GaussianNB().fit(trainx,trainy)
#Evaluating
train_pred=model.predict(trainx)
test_pred=model.predict(testx)
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_acc=accuracy_score(test_pred,testy)
print("Train Acc",train_acc)
print("Test Acc",test_acc)



# for  CATEGORICAL INDEPENDENT VARIABLES
from sklearn.naive_bayes import MultinomialNB
model2=MultinomialNB().fit(trainx,trainy)
#Evaluating
train_pred=model2.predict(trainx)
test_pred=model2.predict(testx)
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_acc=accuracy_score(test_pred,testy)
print("Train Acc",train_acc)
print("Test Acc",test_acc)
