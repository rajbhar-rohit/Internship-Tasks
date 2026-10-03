import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# For Uploading Dataset
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("house_prices.csv")
df.head()

# Exploring the Dataset
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
df.describe()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Area (sq ft)", "Bedrooms", "Age of House (years)"]):
    ax.scatter(df[col], df["Price"], alpha=0.6)
    ax.set_xlabel(col); ax.set_ylabel("Price")
plt.tight_layout(); plt.show()

print(df.corr()["Price"].sort_values(ascending=False))

# Split into Features

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training houses:", len(X_train), "| Test houses:", len(X_test))

# Train two models: Linear Regression and Random Forest Regressor and Compare their performance

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {"MAE": mean_absolute_error(y_test, pred),
                     "R2": r2_score(y_test, pred), "pred": pred}
    print(f"{name}:  MAE = {results[name]['MAE']:,.0f}   R2 = {results[name]['R2']:.3f}")

best_name = max(results, key=lambda n: results[n]["R2"])
print("\nBest model:", best_name)

# Visualizing the results
plt.figure(figsize=(6, 6))
plt.scatter(y_test, results[best_name]["pred"], alpha=0.7)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual price"); plt.ylabel("Predicted price")
plt.title(f"Actual vs Predicted ({best_name})")
plt.legend(); plt.show()

lr = models["Linear Regression"]
print("Linear Regression – effect on price:")
for feat, coef in zip(X.columns, lr.coef_):
    print(f"  +1 {feat:<22} -> {coef:+,.0f}")

rf = models["Random Forest"]
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", title="Random Forest – feature importance")
plt.show()


#Predict the price of a new house

new_house = pd.DataFrame([{
    "Area (sq ft)": 1800,
    "Bedrooms": 3,
    "Age of House (years)": 10,
}])
price = models[best_name].predict(new_house)[0]
print(f"Predicted price: {price:,.0f}")