import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

FEATURES = ["Area (sq ft)", "Bedrooms", "Age of House (years)"]
TARGET = "Price"


# ---------- Data ----------
@st.cache_data
def load_data(path="house_prices.csv"):
    return pd.read_csv(path)


if os.path.exists("house_prices.csv"):
    df = load_data()
else:
    st.warning("`house_prices.csv` not found. Please upload it.")
    up = st.file_uploader("Upload house_prices.csv", type="csv")
    if up is None:
        st.stop()
    df = pd.read_csv(up)

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()


# ---------- Models ----------
@st.cache_resource
def train_models(data):
    X, y = data[FEATURES], data[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        scores[name] = {
            "MAE": mean_absolute_error(y_te, pred),
            "R2": r2_score(y_te, pred),
            "pred": pred,
        }
    return models, scores, y_te


models, scores, y_test = train_models(df)
best = max(scores, key=lambda n: scores[n]["R2"])

# ---------- Header ----------
st.title("🏠 House Price Predictor")
st.caption("A mini machine learning project – predict a house's price from its area, bedrooms and age.")

# ---------- Sidebar inputs ----------
st.sidebar.header("Enter house details")
area = st.sidebar.slider("Area (sq ft)", int(df[FEATURES[0]].min()), int(df[FEATURES[0]].max()), 1500, step=10)
bedrooms = st.sidebar.slider("Bedrooms", int(df[FEATURES[1]].min()), int(df[FEATURES[1]].max()), 3)
age = st.sidebar.slider("Age of house (years)", int(df[FEATURES[2]].min()), int(df[FEATURES[2]].max()), 10)
model_name = st.sidebar.selectbox(
    "Model", list(models), index=list(models).index(best),
    help="Linear Regression is the best performer on this dataset."
)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore data", "🧠 Model insights"])

# ---------- Tab 1: Predict ----------
with tab1:
    new_house = pd.DataFrame([{FEATURES[0]: area, FEATURES[1]: bedrooms, FEATURES[2]: age}])
    price = models[model_name].predict(new_house)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted price", f"{price:,.0f}")
    c2.metric("Typical error (MAE)", f"± {scores[model_name]['MAE']:,.0f}")
    c3.metric("Model accuracy (R²)", f"{scores[model_name]['R2']:.2f}")

    st.info(
        f"Estimated range: **{price - scores[model_name]['MAE']:,.0f} – "
        f"{price + scores[model_name]['MAE']:,.0f}** (prediction ± average test error)"
    )

    st.subheader("Compare both models")
    both = pd.DataFrame({
        "Model": list(models),
        "Predicted price": [f"{m.predict(new_house)[0]:,.0f}" for m in models.values()],
        "R²": [round(scores[n]["R2"], 3) for n in models],
        "MAE": [f"{scores[n]['MAE']:,.0f}" for n in models],
    })
    st.dataframe(both, hide_index=True, width="stretch")

# ---------- Tab 2: Explore ----------
with tab2:
    st.subheader("Dataset preview")
    st.write(f"{df.shape[0]} houses × {df.shape[1]} columns")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("Summary statistics")
    st.dataframe(df.describe().round(1), width="stretch")

    st.subheader("Feature vs Price")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, FEATURES):
        ax.scatter(df[col], df[TARGET], alpha=0.5)
        ax.scatter([{FEATURES[0]: area, FEATURES[1]: bedrooms, FEATURES[2]: age}[col]],
                   [price], color="red", s=120, label="Your house", zorder=3)
        ax.set_xlabel(col)
        ax.set_ylabel("Price")
    axes[0].legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Correlation with price")
    st.bar_chart(df.corr()[TARGET].drop(TARGET))

# ---------- Tab 3: Insights ----------
with tab3:
    st.subheader(f"Actual vs Predicted – {model_name}")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.scatter(y_test, scores[model_name]["pred"], alpha=0.7)
    lims = [y_test.min(), y_test.max()]
    ax2.plot(lims, lims, "r--", label="Perfect prediction")
    ax2.set_xlabel("Actual price")
    ax2.set_ylabel("Predicted price")
    ax2.legend()
    st.pyplot(fig2)

    left, right = st.columns(2)
    with left:
        st.subheader("Linear Regression effects")
        coefs = pd.Series(models["Linear Regression"].coef_, index=FEATURES)
        for feat, c in coefs.items():
            st.write(f"**+1 {feat}** → {c:+,.0f}")
    with right:
        st.subheader("Random Forest feature importance")
        st.bar_chart(pd.Series(models["Random Forest"].feature_importances_, index=FEATURES))

    st.caption(
        "⚠️ Trained on only 200 houses and tested on 40, so treat predictions as rough estimates. "
        "Inputs outside the dataset's range are not supported."
    )