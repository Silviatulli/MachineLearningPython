# read the data
from pydataset import data
iris=data('iris')

# split the data
x=iris.iloc[:,:-1]
y=iris.iloc[:,-1]
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2)

#Model creation
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=10)
model.fit(trainx,trainy)
train_pred=model.predict(trainx)
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_pred=model.predict(testx)
test_acc=accuracy_score(test_pred,testy)
print("Train acc",train_acc)
print("Test_acc",test_acc)

#create decision tree using information needed
model=RandomForestClassifier(n_estimators=100,criterion='entropy',n_jobs=2)
model.fit(trainx,trainy)
train_pred=model.predict(trainx)
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_pred=model.predict(testx)
test_acc=accuracy_score(test_pred,testy)
print("Train acc",train_acc)
print("Test_acc",test_acc)
