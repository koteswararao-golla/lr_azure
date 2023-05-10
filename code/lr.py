import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
#data = pd.read_csv('data.csv')
data = pd.DataFrame({'x':[189,132,785,535,545,365,546,145,236,658],'y':[18,14,75,52,54,35,55,15,24,66]})
print('shape: ',data.shape)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['x']], data['y'], test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the R^2 score on the testing set
print('R^2 score:', model.score(X_test, y_test))
