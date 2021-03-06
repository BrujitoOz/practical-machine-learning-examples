#%% dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv("../Social_Network_ads.csv")
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values
#%% split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#%% scale
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#%% train
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(x_train, y_train)
#%% predict
y_pred = classifier.predict(x_test)
print(y_pred)
print(y_test)
#%% matrix
from sklearn .metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
def MatrizdeConfusion(cm):
    print(pd.DataFrame({' ':['No', 'Si'],'No':cm[0],'Si':cm[1]}))
    aciertos = 0
    desaceritos = 0
    for i in range(len(cm[0])):
        aciertos += cm[i][i]
    desaceritos = sum(sum(cm))
    desaceritos -= abs(aciertos) 
    print("Number of hits: ",aciertos)
    print("Number of mistakes: ",desaceritos)
MatrizdeConfusion(cm)

#%% view
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Classifier (Training Set) ')
plt.xlabel('age')
plt.ylabel('Estimated Salary ')
plt.legend()
plt.show()

X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ['red', 'green'][i], label = j)
plt.title('Classifier (test Set) ')
plt.xlabel('age')
plt.ylabel('Estimated Salary ')
plt.legend()
plt.show()