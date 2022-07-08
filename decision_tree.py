# Reading the dataset
from pydataset import data
iris=data('iris')
iris.head()

iris.isna().sum()
# splitting the dataset into train and test
x=iris.drop('Species',axis=1)
y=iris['Species']

from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=0.2)

print("Trainx size",trainx.shape)
print("Trainy size",trainy.shape)
print("Testx size",testx.shape)
print("Testy size",testy.shape)

# Model creation --- default gini
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(trainx,trainy)

# predict using model
train_pred=model.predict(trainx)
test_pred=model.predict(testx)

# calculate accuracy
from sklearn.metrics import accuracy_score
train_acc=accuracy_score(train_pred,trainy)
test_acc=accuracy_score(test_pred,testy)
print("Training acc",train_acc)
print("Testing acc",test_acc)

# Model2 creation --- using information gain
from sklearn.tree import DecisionTreeClassifier
model_new=DecisionTreeClassifier(criterion='entropy')
model_new.fit(trainx,trainy)

# predict using model
train_pred=model_new.predict(trainx)
test_pred=model_new.predict(testx)
train_acc=accuracy_score(train_pred,trainy)
test_acc=accuracy_score(test_pred,testy)
print("Training acc",train_acc)
print("Testing acc",test_acc)

## Plot
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(20, 10))
from sklearn import tree
plot = tree.plot_tree(model, 
                   feature_names=iris.columns,  
                   class_names=trainy.unique(),
                   filled=True)
