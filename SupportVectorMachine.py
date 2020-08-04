# Import relevant libraries
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# Load data
cancer = datasets.load_breast_cancer()
"""print(cancer.feature_names)
print(cancer.target_names)"""

# Variables
X = cancer.data
y = cancer.target

# Split model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.2)

print(X_train, y_train)
classes = ['malignant' 'benign']

# Classifier
clf = svm.SVC(kernel = "linear", C=1)
#clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)