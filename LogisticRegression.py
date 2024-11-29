import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import random

data = pd.read_excel("ForestCover.xlsx")
# print(data.describe())
# print(data.info())

dummies = pd.get_dummies(data, columns=['States/UTs'], dtype='int64')
# print(dummies)
data1 = dummies.drop(columns=['ID'])
# print(data1)
# print(data1.corr().to_string())

x = data1[['2011 Assessment - VDF']]
y_continuous = data1[['2011 Assessment - Total']]
threshold = y_continuous.mean()  # You can set your own threshold
y = (y_continuous > threshold).astype(int)

random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
model = LogisticRegression()
model.fit(x_train, y_train)

y_pre = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pre)
print("Model Score : ", model.score(x_train, y_train))
print("Accuracy : {:.2f}%".format(accuracy * 100))

# Calculate regression scores
mse = mean_squared_error(y_test, y_pre)
mae = mean_absolute_error(y_test, y_pre)
r2 = r2_score(y_test, y_pre)
evs = explained_variance_score(y_test, y_pre)

# Print the scores
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)
print("Explained Variance Score (EVS):", evs)

# Additional score: Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)


# Visualizing the relationship between features and target
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data1['2011 Assessment - VDF'], y=data1['2011 Assessment - Total'])
plt.xlabel('2011 Assessment - VDF')
plt.ylabel('2011 Assessment - Total')
plt.title('2011 Assessment - VDF vs 2011 Assessment - Total')
plt.show()

# Visualizing predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pre)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.axhline(0.5, color='red', linestyle='--')  # Add a line for the threshold
plt.show()
