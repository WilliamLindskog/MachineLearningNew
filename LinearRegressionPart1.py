# Import all relevant libraries
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

# Read data
data = pd.read_csv('student-mat.csv', sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#What we seek to predict
predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size= 0.1)

"""# Train and test samples
best = 0
for _ in range(100):

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size= 0.1)

    # Linear Model
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    acc = reg.score(X_test,y_test)
    print(acc)

    if acc > best:
        best = acc
        # Save a pickle file in my directory
        with open("Studentmodel.pickle", "wb") as f:
            pickle.dump(reg, f)
"""
# Read our pickle-file
pickle_input = open("Studentmodel.pickle", "rb")
reg = pickle.load(pickle_input)

# Viewing the constants
print('Coefficient: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# Predictions
predictions = reg.predict(X_test)
sum_diff = 0
sum = []
for x in range(len(predictions)):
    print("Iteration:", x + 1, "\nPrediction:", predictions[x], "\nX variables:", X_test[x], "\nTest value:",  y_test[x], "\nDifference:", predictions[x] - y_test[x], "\n")
    sum_diff += abs(predictions[x] - y_test[x])
    sum.append(sum_diff)
print("Total sum of difference:", sum_diff)
print("Average difference per iteration:", sum_diff/len(predictions))
# print("Accuracy:", best)
p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final grade")
plt.show()
"""plt.plot(sum)
plt.show()"""
