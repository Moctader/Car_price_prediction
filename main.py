import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


#Data collection and processing

# Loading the data from csv file to pandas data frame
car_dataset = pd.read_csv('/Users/moctader/Downloads/archive/car data.csv')
print(car_dataset.head(7))
car_dataset.shape

# Getting some information about this dataset
print(car_dataset.info())

# Check the number of missing value
print(car_dataset.isnull().sum())

# Check the distribution of the categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# Encoding the categorical data
# Encoding the "Fuel_Type" column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG': 2}}, inplace= True)

# Encoding the "Seller_Type" column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace= True)

# Encoding the "Transmission" column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}}, inplace= True)


# Splitting the data into test and train
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis =1)
Y = car_dataset['Selling_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.1, random_state=2)

# Model Training

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

#Model Evaluation
# Prediction on training data
prediction_training_data = lin_reg_model.predict(X_train)

# R square error
error_score = metrics.r2_score(Y_train, prediction_training_data)
print('R squared erroe: ',error_score)

# visualize the actual price and the predicted price
plt.figure(1)
plt.scatter(Y_train, prediction_training_data)
plt.xlabel('Actual price')
plt.ylabel('predicted prices')
plt.title('Actual price vs Predicted prices')
plt.show()

# Prediction on test data
prediction_test_data = lin_reg_model.predict(X_test)

# R square error
error_score2 = metrics.r2_score(Y_test, prediction_test_data)
print('R squared erroe: ',error_score2)


# visualize the actual price and the predicted price for test data
plt.figure(2)
plt.scatter(Y_test, prediction_test_data)
plt.xlabel('Actual price')
plt.ylabel('predicted prices')
plt.title('Actual price vs Predicted test prices')
plt.show()

# 2. Lasso Regression model prediction

lass_reg_model = Lasso()
lass_reg_model.fit(X_train, Y_train)

#Model Evaluation
# Prediction on training data
prediction_training_data = lass_reg_model.predict(X_train)

# R square error
error_score = metrics.r2_score(Y_train, prediction_training_data)
print('R squared erroe: ',error_score)

# visualize the actual price and the predicted price
plt.figure(3)
plt.scatter(Y_train, prediction_training_data)
plt.xlabel('Actual price')
plt.ylabel('predicted prices')
plt.title('Actual price vs Predicted prices')
plt.show()

# Prediction on test data
prediction_test_data = lass_reg_model.predict(X_test)

# R square error
error_score2 = metrics.r2_score(Y_test, prediction_test_data)
print('R squared erroe: ',error_score2)


# visualize the actual price and the predicted price for test data
plt.figure(4)
plt.scatter(Y_test, prediction_test_data)
plt.xlabel('Actual price')
plt.ylabel('predicted prices')
plt.title('Actual price vs Predicted test prices')
#plt.show()
