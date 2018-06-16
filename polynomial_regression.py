# import libraries
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# import dataset
dataset = pandas.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values
X = X.reshape(-1,1)
y = dataset.iloc[:, 2].values
y = y.reshape(-1,1)

# fitting the polynomial regression to the dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression().fit(X_poly, y)

# visualising the dataset
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Salary vs Position Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()