import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reading the dataset
data = "moisture_days.csv"


# Create a DataFrame from the data
df = pd.read_csv(data)

# Extracting features and target variable
X = df[['days']]  # Features
y = df['moisture']  # Target variable

# Encoding categorical variable 'crop'
#X_encoded = pd.get_dummies(X, columns=['moisture'])

# Splitting the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Scaling numerical features ('temp' column)
#scaler = StandardScaler()
#X_train['moisture'] = scaler.fit_transform(X_train[['moisture']])
#X_test['moisture'] = scaler.transform(X_test[['moisture']])

# Initializing and training the Random Forest Regressor model
#model = RandomForestRegressor(random_state=42)
#model.fit(X_train, y_train)

# Predicting on the test set
#y_pred = model.predict(X_test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
