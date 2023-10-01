import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# The data is provided by The Numbers, and it contains over 5000 movies between 1915 and 2017.
data = pandas.read_csv('cost_revenue_clean.csv')

# feature / independant value
X = DataFrame(data, columns=['production_budget_usd'])
# target / dependant value
y = DataFrame(data, columns=['worldwide_gross_usd'])

# A residual is the gap (or difference) between the actual y-value and the predicted (fitted) y-value.
# To find the best possible theta_0 and theta_1, we pick the line with the lowest Residual Sum of Squares (RSS).
regression = LinearRegression()
regression.fit(X, y)


def displayGraph(drawLinearRegressionLine):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.3)
    # Adding the regression line here:
    if drawLinearRegressionLine == True:
        plt.plot(X, regression.predict(X), color='red', linewidth=1)
    plt.title('Film Cost vs Global Revenue')
    plt.xlabel('Production Budget $')
    plt.ylabel('Worldwide Gross $')
    plt.ylim(0, 3000000000)
    plt.xlim(0, 450000000)
    plt.show()


print(data.describe())
print(f'Theta 0 = {regression.intercept_}')  # theta_0 / y-intercept
print(f'Theta 1 = {regression.coef_}')  # theta_1 / slope
print(f'R-Square = {regression.score(X, y)}')  # r-square
displayGraph(drawLinearRegressionLine=False)  # original graph
displayGraph(drawLinearRegressionLine=True)  # graph with LinearRegressionLine
