import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

df = pd.read_excel('dataset6.xlsx')

# Split the data into X and y
X = df.iloc[:, 1:12].values
y = df.iloc[:,-1].values
y = y.reshape(-1, 1)

# Veri setinin eğitim ve test olarak bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

import xlsxwriter

workbook = xlsxwriter.Workbook('X_train.xlsx')
worksheet = workbook.add_worksheet()

col = 0

for row, data in enumerate(X_train):
    worksheet.write_row(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('X_test.xlsx')
worksheet = workbook.add_worksheet()

col = 0

for row, data in enumerate(X_test):
    worksheet.write_row(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('y_train.xlsx')
worksheet = workbook.add_worksheet()

col = 0

for row, data in enumerate(y_train):
    worksheet.write_row(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('y_test.xlsx')
worksheet = workbook.add_worksheet()

col = 0

for row, data in enumerate(y_test):
    worksheet.write_row(row, col, data)

workbook.close()

# Özellik ölçekleme
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#sc_y = MinMaxScaler()
#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.fit_transform(y_test)

# Eğitim verileri ile modelin eğitilmesi
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Test verileri kullanılarak tahmin yapılması
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#y_test_non_transofrmed = sc_y.inverse_transform(y_test)
#y_pred_non_transformed = sc_y.inverse_transform(y_pred)

import matplotlib.pyplot as plt

plt.scatter(y_test,y_pred)
plt.xlabel("Test Verileri") 
plt.ylabel("Tahmin Verileri")
plt.show()

plt.plot(y_test,color = 'blue')
plt.plot(y_pred, color = 'orange')
plt.show()

# Print the metrics
print('\nMAE:', mae)
print('MSE:', mse)
print('R^2:', r2)
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
