import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel('dataset4.xlsx')

# Split the data into X and y
X = df.iloc[:, 1:12].values
y = df.iloc[:,-1].values
y = y.reshape(-1, 1)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:,3:4])
X[:, 3:4] = imputer.transform(X[:,3:4])

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 2] = labelencoder.fit_transform(X[:, 2])

# Veri setinin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

import xlsxwriter

workbook = xlsxwriter.Workbook('X_train.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(X_train):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('X_test.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(X_test):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('y_train.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(y_train):
    worksheet.write_column(row, col, data)

workbook.close()

workbook = xlsxwriter.Workbook('y_test.xlsx')
worksheet = workbook.add_worksheet()

row = 0

for col, data in enumerate(y_test):
    worksheet.write_column(row, col, data)

workbook.close()
# Özellik ölçekleme
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)

# Eğitim verileri ile modelin eğitilmesi
model = LinearRegression()
model.fit(X_train, y_train)

# Test verileri kullanılarak tahmin yapılması
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

"""plt.scatter(y_test,y_test)
plt.show()
plt.scatter(y_pred.index,y_pred)
plt.show()"""
plt.scatter(y_test,y_pred)
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
