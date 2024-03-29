import pandas as pd
# Data Gathering
salary=pd.read_csv('SalaryData.csv')
salary.head()

# Data Processing
from sklearn.preprocessing import LabelEncoder
cat_col=[]
for i in salary.columns:
    if salary[i].dtype=='object':
        cat_col.append(i)
        

for i in cat_col:
    var=i+'_lbl'
    String = "var%s = %s"%(var, 'LabelEncoder()') 
    exec(String)
    var=LabelEncoder()
    salary[i]=var.fit_transform(salary[i])
salary.head()

salary.describe()

# Normalization
con_col=['capitalgain','capitalloss','hoursperweek','native']
from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler()
for i in con_col:
    salary[i]=norm.fit_transform(salary[i].values.reshape(-1, 1))

# Train and Test Split
x=salary.drop('Salary',axis=1)
y=salary['Salary']
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(x,y,test_size=.2)
print(trainx.shape)
print(testx.shape)

# Model Creation SVM  # default - rbf kernel
from sklearn.svm import SVC
model1=SVC().fit(trainx,trainy)
model1_train_pred=model1.predict(trainx)
model1_test_pred=model1.predict(testx)

# Model Evaluation SVM  # default - rbf kernel
from sklearn.metrics import accuracy_score
print("Train acc",accuracy_score(model1_train_pred,trainy))
print("Train acc",accuracy_score(model1_test_pred,testy))

model1.get_params()

# Model Creation SVM - linear kernel
model_linear=SVC(kernel='linear').fit(trainx,trainy)
model_linear_train_pred=model_linear.predict(trainx)
model_linear_test_pred=model_linear.predict(testx)

# Model Evaluation SVM - linear kernel
from sklearn.metrics import accuracy_score
print("Train acc",accuracy_score(model_linear_train_pred,trainy))
print("Train acc",accuracy_score(model_linear_test_pred,testy))

# Model Creation SVM - sigmoid kernel
model_sigmoid=SVC(kernel='sigmoid').fit(trainx,trainy)
model_sigmoid_train_pred=model_sigmoid.predict(trainx)
model_sigmoid_test_pred=model_sigmoid.predict(testx)

# Model Evaluation SVM - sigmoid kernel
from sklearn.metrics import accuracy_score
print("Train acc",accuracy_score(model_sigmoid_train_pred,trainy))
print("Train acc",accuracy_score(model_sigmoid_test_pred,testy))

