import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import seaborn as sns

warnings.filterwarnings('ignore')

df=pd.read_csv("C:/Users/Dragonoid/Desktop/6th Semester/Pattern/Project/movies-regression/movies-regression-dataset.csv")



Y=df['vote_average']
X=df.loc[:, df.columns != 'vote_average']



from sklearn.naive_bayes import GaussianNB

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create a Naive Bayes model
nb_model = GaussianNB()

# Train the Naive Bayes model
nb_model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = nb_model.predict(x_test)

# Evaluate the model
accuracy = nb_model.score(x_test, y_test)
print("Accuracy:", accuracy)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # specify the number of trees (n_estimators) in the forest
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse3 = mean_squared_error(y_test, y_pred)
print('Random Forest MSE:', mse3)

# Scatter plot of actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)

# Plot a line of perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')

# Set the labels and title
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Random Forest Regression')
plt.show()