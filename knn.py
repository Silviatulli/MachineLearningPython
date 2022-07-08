## KNN
# Loading data set
from pydataset import data
iris=data('iris')


x=iris.drop('Species',axis=1)
y=iris['Species']

#splitting the dataset
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2)

#KNN model creation
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=7)
model.fit(trainx,trainy)

#Evaluating
train_pred=model.predict(trainx)
test_pred=model.predict(testx)

from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_acc=accuracy_score(test_pred,testy)

print("Train Acc",train_acc)
print("Test Acc",test_acc)

#Finding K value
acc=[]
for i in range(1,25,2):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(trainx,trainy)
    train_pred=model.predict(trainx)
    test_pred=model.predict(testx)
    train_acc=accuracy_score(train_pred,trainy)
    test_acc=accuracy_score(test_pred,testy)
    acc.append([train_acc,test_acc,i])
    
#matplotlib auto
import matplotlib.pyplot as plt
plt.plot(range(1,25,2),[i[0] for i in acc],c='b')
plt.plot(range(1,25,2),[i[1] for i in acc],c='r')
plt.legend(['Train','Test'])
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()


# Regression problem
# Dependent variable(y) should be continues
from sklearn.neighbors import KNeighborsRegresso
