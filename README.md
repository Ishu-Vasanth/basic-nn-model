# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. These units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![OP-01](https://github.com/Ishu-Vasanth/basic-nn-model/assets/94154614/0c64b6a9-7538-4bbe-8f2c-b81ae99998ff)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

# PROGRAM
```
Developed By: ISHWARYA V 
RegNo: 212221240016
```
## Importing Required Packages :
```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth

```
## Authentication and Creating DataFrame From DataSheet:
```

import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl').sheet1
data = worksheet.get_all_values()
dataset = pd.DataFrame(data[1:], columns=data[0])
dataset = dataset.astype({'Input':'float'})
dataset = dataset.astype({'Output':'float'})
dataset.head()
```
## Assigning X and Y values :
```

X = dataset[['Input']].values
Y = dataset[['Output']].values
```
## Normalizing the data :
```

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.33,random_state = 20)
Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train_scale = Scaler.transform(x_train)
```
## Creating and Training the model :
```

my_brain = Sequential([
    Dense(units = 4, activation = 'relu' , input_shape=[1]),
    Dense(units = 6),
    Dense(units = 1)

])
my_brain.compile(optimizer='rmsprop',loss='mse')
my_brain.fit(x=x_train_scale,y=y_train,epochs=20000)
```
## Plot the loss :
```

loss_df = pd.DataFrame(my_brain.history.history)
loss_df.plot()
```
## Evaluate the Model :
```

x_test1 = Scaler.transform(x_test)
my_brain.evaluate(x_test1,y_test)
```
## Prediction for a value :
```

X_n1 = [[30]]
input_scaled = Scaler.transform(X_n1)
my_brain.predict(input_scaled)
```












## Dataset Information
![OP-2](https://github.com/Ishu-Vasanth/basic-nn-model/assets/94154614/bc0bcdd9-a58a-48eb-a937-8de39f896fc5)


## OUTPUT

### Training Loss Vs Iteration Plot

![OP-3](https://github.com/Ishu-Vasanth/basic-nn-model/assets/94154614/e7e4a36b-4856-4cdb-966f-8c479f24413a)


### Test Data Root Mean Squared Error

![OP-4](https://github.com/Ishu-Vasanth/basic-nn-model/assets/94154614/15644ac0-d42a-4995-9c74-2cbd198f1a33)


### New Sample Data Prediction

![OP-5](https://github.com/Ishu-Vasanth/basic-nn-model/assets/94154614/e23dcc9b-f571-41b9-91bb-fe80280e4d27)


# RESULT:
Therefore We successfully developed a neural network regression model for the given dataset.


