import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
import statsmodels.api as sm
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_excel('dataset.xlsx')

X = df.iloc[:, 1:12].values
y = df.iloc[:,-1].values
y = y.reshape(-1, 1) 

#encoding
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])

best_ratio = 0.3
scaler = 1

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=best_ratio)
X_train = pd.read_excel('./Best/X_train.xlsx').values
X_test = pd.read_excel('./Best/X_test.xlsx').values
y_train = pd.read_excel('./Best/y_train.xlsx').values
y_test = pd.read_excel('./Best/y_test.xlsx').values


#feature scaling - 1 for minmax 0 for S.S
if(scaler == 1):
    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
else:
    sc_X = StandardScaler()
    sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#y_train = sc_y.fit_transform(y_train)
#y_test = sc_y.transform(y_test)

#applying MLR
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
"""
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
"""
#metric calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

#print the metrics
print('\nMAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R^2:', r2)

coefficients = model.coef_

#print the pie chart
sizes = abs(model.coef_.flatten())
labels = df.columns[1:-1]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown','gray'] 

plt.pie(sizes, labels=labels, colors=colors)
plt.title("Coefficients")
plt.show()

#print the bar graph
plt.bar(labels,sizes)
plt.ylabel('Coefficients')
plt.xticks(rotation=60,fontsize=10)
plt.show()


fitted_values = model.predict(X_train)
residuals = y_train - fitted_values
plt.scatter(fitted_values, residuals,s=10)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Fark')
plt.show()

plt.scatter(y_train, fitted_values,s=10)
plt.xlabel('Ger??ek De??er')
plt.ylabel('Tahmin Edilen')
plt.show()


plt.plot(y_test,color = 'blue')

plt.plot(y_pred, color = 'orange')
plt.show()

residuals = y_train - fitted_values

sm.qqplot(residuals, line='s')
plt.show()

sns.pairplot(df)
plt.show()

#svr,neuralnetwork,ayn?? ds,
workbook = xlsxwriter.Workbook('.\saved\X_train.xlsx')
worksheet = workbook.add_worksheet()
col = 0
for row, data in enumerate(X_train):
    worksheet.write_row(row, col, data)
workbook.close()


workbook = xlsxwriter.Workbook('.\saved\X_test.xlsx')
worksheet = workbook.add_worksheet()
col = 0
for row, data in enumerate(X_test):
    worksheet.write_row(row, col, data)
workbook.close()


workbook = xlsxwriter.Workbook('.\saved\y_train.xlsx')
worksheet = workbook.add_worksheet()
col = 0
for row, data in enumerate(y_train):
    worksheet.write_row(row, col, data)
workbook.close()


workbook = xlsxwriter.Workbook('.\saved\y_test.xlsx')
worksheet = workbook.add_worksheet()
col = 0
for row, data in enumerate(y_test):
    worksheet.write_row(row, col, data)    
workbook.close()









#algorithm to find best train-test split
"""
best_r2 = 0
loop = 500
tsize = [0.2,0.25,0.3,0.35,0.4]
print(tsize)

#finding optimum train-test ratio
for a in tsize:
    r2 = 0
    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=a)
        sc_X = MinMaxScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)

        sc_y = MinMaxScaler()
        y_train = sc_y.fit_transform(y_train) 
        y_test = sc_y.fit_transform(y_test)
        
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
            
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)
        y_test = sc_y.transform(y_test)
        


        
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)  

        r2 += r2_score(y_test, y_pred)

    print(r2/loop) 
    r2 /= loop

    if r2 > best_r2:
        best_ratio = a
        best_r2 = r2

print("\nBest train-test ratio:", best_ratio)
print("Best R2 score:", best_r2)
"""
