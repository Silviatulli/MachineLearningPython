#Logistic Regression

# Data Gathering
from pydataset import data
titanic=data('titanic')
titanic1=data('titanic')

# Exploratory Data Analysis
titanic.columns
titanic['survived'].value_counts()
titanic['class'].nunique()
titanic['class'].unique()
titanic['age'].unique()
titanic['sex'].unique()


# Data Processing
from sklearn.preprocessing import LabelEncoder

class_lbl=LabelEncoder()
titanic['class']=class_lbl.fit_transform(titanic['class'])

age_lbl=LabelEncoder()
titanic['age']=age_lbl.fit_transform(titanic['age'])

sex_lbl=LabelEncoder()
titanic['sex']=sex_lbl.fit_transform(titanic['sex'])


## Train and Test Splitting
x=titanic.iloc[:,:-1]
y=titanic.iloc[:,-1]
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2,random_state=0)


# Model Creation
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(trainx,trainy)

# Model Evaluation
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
