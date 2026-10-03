# 🏠 House Price Prediction – Mini AI Project

A beginner-friendly machine learning project that predicts a house's **price** from its **area**, **number of bedrooms** and **age**. It is written as a single Python script that runs in **Google Colab**, so nothing needs to be installed.

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `house_price_prediction.ipynb` / `.py` | The code (Colab notebook or script) |
| `house_prices.csv` | Dataset of 200 houses |
| `README.md` | This file |

---

## 📊 Dataset

`house_prices.csv` has **200 rows × 4 columns** and **no missing values**.

| Column | Role | Description |
|--------|------|-------------|
| `Area (sq ft)` | Feature | Size of the house |
| `Bedrooms` | Feature | Number of bedrooms (1–5) |
| `Age of House (years)` | Feature | Age of the house (0–40) |
| `Price` | **Target** | Price we want to predict |

All columns are numeric, so no encoding or scaling is needed.

---

## 🚀 How to Run (Google Colab)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Paste the code (or upload the `.ipynb` file).
3. Run the cell(s). When the upload prompt appears, click **Choose files** and select `house_prices.csv`.

> ⚠️ **Keep the filename exactly `house_prices.csv`.** If you upload twice, Colab renames the second copy (e.g. `house_prices (1).csv`) and `pd.read_csv("house_prices.csv")` will still read the old one.

---

## 🧠 Code Walkthrough

| Step | What the code does | Key idea |
|------|--------------------|----------|
| **1. Imports** | Loads pandas, NumPy, Matplotlib and scikit-learn tools | Libraries for data, plots and ML |
| **2. Upload data** | `files.upload()` then `pd.read_csv(...)` | Colab-only file upload |
| **3. Explore** | Prints shape and missing values, draws 3 scatter plots, prints correlations with `Price` | Understand data before modelling |
| **4. Split** | `X` = features, `y` = `Price`; `train_test_split(test_size=0.2, random_state=42)` | 160 houses to train, 40 to test |
| **5. Train two models** | Fits `LinearRegression` and `RandomForestRegressor(n_estimators=200)` in a loop, scores each with MAE and R² | Compare a simple and a complex model |
| **6. Pick the best** | `max(results, key=...)` chooses the model with the highest R² | Automatic model selection |
| **7. Visualize** | Actual vs predicted scatter plot, linear regression coefficients, random forest feature importance | Interpret what the models learned |
| **8. Predict** | Builds a one-row DataFrame for a new house and predicts its price | Using the trained model |

---

## 📈 Results

| Model | MAE (avg error) | R² |
|-------|----------------:|---:|
| **Linear Regression** ✅ | ~9,748 | 0.941 |
| Random Forest | ~10,612 | 0.916 |

**Correlation with Price:** Area 0.93 · Bedrooms 0.91 · Age −0.29

**What Linear Regression learned**

- +1 sq ft of area → about **+103** in price
- +1 bedroom → about **+3,563**
- +1 year of age → about **−1,025**

**Sample prediction:** a 1,800 sq ft, 3-bedroom, 10-year-old house is predicted at roughly **209,759**.

Linear Regression wins here because the data is small and its relationships are close to straight lines, which suits a linear model well.

---

## 📖 Key Concepts

- **Features (X) and target (y):** the inputs we know and the value we predict.
- **Train/test split:** the model learns on 80% of the data and is graded on the unseen 20%.
- **`random_state=42`:** makes the split repeatable, so you get the same results each run.
- **Linear Regression:** fits a straight-line formula, easy to interpret.
- **Random Forest:** averages many decision trees, good for complex patterns.
- **MAE:** average size of the prediction error (lower is better).
- **R²:** share of price variation explained (1.0 is perfect).

---

## ⚠️ Notes and Limitations

- **Only 40 test houses**, so scores can shift noticeably with a different split. Cross-validation gives a more reliable estimate.
- **Area and bedrooms are correlated** (bigger houses have more bedrooms), so individual coefficients should be read as rough guides, not exact effects.
- **Stay within the data's range.** Predictions for houses far outside 492–2,425 sq ft or 0–40 years of age are unreliable, especially for Random Forest.
- **In Colab, only the last expression in a cell is displayed.** If you put all the code in one cell, `df.head()` and `df.describe()` will show nothing unless wrapped in `print(...)`, or placed in separate cells.

---


## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · scikit-learn · Google Colab
