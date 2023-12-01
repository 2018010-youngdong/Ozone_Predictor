import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

filename = './data/airquality.csv'
data=pd.read_csv(filename)

print(data.isnull().sum())
data = data.dropna(subset=['Ozone', 'Solar.R'])

correlation_matrix = data.corr()
correlation_ozone = correlation_matrix['Ozone']
plt.figure(figsize=(10, 6))
correlation_ozone.drop('Ozone').plot(kind='bar', color='skyblue')
plt.title("Ozone_Correlation")
plt.xlabel("x")
plt.ylabel("Correlation")
plt.show()

X = data[['Temp', 'Wind']]
y = data['Ozone']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

plt.scatter(y_test, y_pred)
plt.xlabel("true Ozone")
plt.ylabel("test Ozone")
plt.title("test_true value")
plt.show()

