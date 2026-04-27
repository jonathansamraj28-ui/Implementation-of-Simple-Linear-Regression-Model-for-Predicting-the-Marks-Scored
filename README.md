# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Load the dataset and separate input and output variables.
2. Split the data into training and testing sets.
3. Train the linear regression model using the training data.
4. Predict the output for the test data and evaluate the results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jonathan samraj A
RegisterNumber:  212225040160
*/

```
Program to implement the simple linear regression model for predicting the marks scored.
   Developed by: Dhiren D
RegisterNumber: 25007814


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
 
Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="blue")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)


## Output:
<img width="1045" height="687" alt="image" src="https://github.com/user-attachments/assets/20f7a693-3f01-411a-a483-077cd96dc3c6" />
<img width="1050" height="685" alt="image" src="https://github.com/user-attachments/assets/8feed488-ed10-4291-a98a-bb18b3aaf142" />
<img width="260" height="83" alt="image" src="https://github.com/user-attachments/assets/1072ed48-4592-4f8f-9670-b9ba176007f5" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
