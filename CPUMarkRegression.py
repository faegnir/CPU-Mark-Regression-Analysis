import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('dataset3.xlsx')


# Split the data into X and y
X = df.iloc[:, 1:13].values
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

best_ratio = 0
best_r2 = 0
loop = 500
tsize = [0.2,0.25,0.3,0.35,0.4]

print(tsize)


for a in tsize:
    r2 = 0
    for i in range(loop):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=a)

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


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=best_ratio)



sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train) 
y_test = sc_y.fit_transform(y_test)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print the metrics
print('\nMAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('R^2:', r2)