#Logistic Regression
#Loading the data
from pydataset import data
titanic=data('titanic')
titanic1=data('titanic')
#EDA
titanic.columns
titanic['survived'].value_counts()
titanic['class'].nunique()
titanic['class'].unique()
titanic['age'].unique()
titanic['sex'].unique()


#Converting the string values into numeric
from sklearn.preprocessing import LabelEncoder

class_lbl=LabelEncoder()
titanic['class']=class_lbl.fit_transform(titanic['class'])

age_lbl=LabelEncoder()
titanic['age']=age_lbl.fit_transform(titanic['age'])

sex_lbl=LabelEncoder()
titanic['sex']=sex_lbl.fit_transform(titanic['sex'])


## Splitting the data into 4
x=titanic.iloc[:,:-1]
y=titanic.iloc[:,-1]
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2,random_state=0)


#Logistic regression model creation
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(trainx,trainy)

# Train model valuation
train_pred=model.predict(trainx)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
confusion_matrix(trainy,train_pred)
#(545+260)/(545+110+137+260)
accuracy_score(trainy,train_pred)

test_pred=model.predict(testx)
confusion_matrix(testy,test_pred)
accuracy_score(testy,test_pred)

a=classification_report(testy,test_pred)
print(a)

### Testing a new data
import pandas as pd
newdf=pd.DataFrame({'class':['1st class'], 'age':['adults'],  'sex':['man']})
newdf['class']=class_lbl.transform(newdf['class'])
newdf['age']=age_lbl.transform(newdf['age'])
newdf['sex']=sex_lbl.transform(newdf['sex'])
model.predict(newdf)
