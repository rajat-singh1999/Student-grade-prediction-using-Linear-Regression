import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "internet", 'romantic']]
# print(data.head())
data.replace('yes', 1, inplace=True)
data.replace('no', 0, inplace=True)
data.fillna(-99999)
print(data)

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# best = 0
# while best <= 0.95:
#
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#
#     acc = linear.score(x_test, y_test)
#
#     print(acc)
#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# acc = linear.score(x_test, y_test)
#
# print(acc)

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)  # THis value is pre-given
p_array = np.array([20, 19, 1, 0, 3, 1, 1])  # I have put this value myself to show that it can take any input
p_array = p_array.reshape(1, -1)
p = linear.predict(p_array)
print(p)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# The following block is to visualize any two attributes:

# p = 'internet'
# q = 'G3'
# style.use("ggplot")
# pyplot.scatter(data[p], data[q])
# pyplot.xlabel(p)
# pyplot.ylabel("Final Grade")
# pyplot.show()
