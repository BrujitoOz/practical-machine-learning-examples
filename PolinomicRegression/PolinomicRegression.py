#%% dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("../Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#%% linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x, y)
#%% Corr
import seaborn as sns
sns.pairplot(dataset)
print(dataset.corr())
#%% fit polynomial
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(9)
x_poly = poly_reg.fit_transform(x)
lm2 = LinearRegression()
lm2.fit(x_poly, y)
y_pred = lm2.predict(x_poly)
print(pd.DataFrame({'Prediction':np.around(y_pred,2), 'True': y}))
#%% Metric
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("MSA:", mean_absolute_error(y, y_pred))
#%% Ply Regression 
xlabel = "Employee position"
ylabel = "Salary (in $) "
plt.scatter(x, y, color = "red")
plt.plot(x, lm2.predict(x_poly), color = "blue")
plt.title("Polynomial regression model")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
# Regresion polinomica suavizada
x_grid = np.arange(min(x), 10.1, 0.1)
x_grid = x_grid.reshape(len(x_grid) ,1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, lm2.predict(  poly_reg.fit_transform(x_grid) ), color = "blue")
plt.title("Smoothed Polynomial Regression Model")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
# %%
