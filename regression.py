from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

df = load_data()
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42)
}

params = {
    "DecisionTree": {
        'max_depth': [2, 4, 6],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "RandomForest": {
        'n_estimators': [50, 100],
        'max_depth': [4, 6],
        'min_samples_split': [2, 5]
    }
}

for name, model in models.items():
    print(f"Hyperparameter tuning for {name}...")
    grid = GridSearchCV(model, params[name], cv=3, scoring='r2')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    mse, r2 = evaluate_model(best_model, X_test, y_test)
    print(f"{name} Best - MSE: {mse:.2f}, R²: {r2:.2f}")
from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

df = load_data()
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    print(f"{name} - MSE: {mse:.2f}, R²: {r2:.2f}")
