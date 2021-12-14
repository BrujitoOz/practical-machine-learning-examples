#%% dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv("../Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#%% scale 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape(-1,1))
#%% fit
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(x,y)
#%% predict 
y_pred = sc_y.inverse_transform(regression.predict((x)))
#%% metric
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE:", mean_squared_error(sc_y.inverse_transform(y), y_pred))
print("RMSE:", np.sqrt(mean_squared_error(sc_y.inverse_transform(y), y_pred)))
print("MSA:", mean_absolute_error(sc_y.inverse_transform(y), y_pred))
#%% view
import matplotlib.pyplot as plt
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = "red")
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regression.predict(x_grid)), color = "blue")
plt.title("Regression model SVR ")
plt.xlabel("Employee position ")
plt.ylabel("Salary (in $) ")
plt.show()
# %%
