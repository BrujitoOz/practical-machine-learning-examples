#%% read
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("../Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#%% split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3, random_state=0)
#%% y = b0 + x1 * b1
from sklearn.linear_model import LinearRegression
MyRegression = LinearRegression()
MyRegression.fit(x_train, y_train)
#%% Interpret coefficients b0 intercept b1 coefficient 
print(MyRegression.intercept_)
print(MyRegression.coef_)
#%% Predict the test set 
import numpy as np
y_pred = MyRegression.predict(x_test)
print(pd.DataFrame({'prediccion': np.around(y_pred), 'real': y_test  }))
#%% Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MSA:", mean_absolute_error(y_test, y_pred))
#%% View training results 
from sklearn.metrics import r2_score
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, MyRegression.predict(x_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Entrenamiento) - R2: {}".format(round(MyRegression.score(x_train, y_train),3)))
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()
# View test  results 
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, MyRegression.predict(x_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Test) - R2: {}".format(round(r2_score(y_pred, y_test),3)))
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo en $")
plt.show()
# %%
