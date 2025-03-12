import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit, fsolve

# ✅ Load Data
@st.cache_data
def load_data():
    file_path = "PP_cleaned.xlsx"
    return pd.read_excel(file_path, sheet_name="Sheet1")

df = load_data()

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Curve Fitting", "ANN Model", "Code Explorer"])

# ✅ Exploratory Data Analysis (EDA)
if page == "EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.write("This section explores the dataset, visualizes distributions, and checks correlations.")

    if st.button("Show Data Summary"):
        st.write(df.describe())

    if st.button("Show Correlation Heatmap"):
        numeric_cols = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        st.pyplot(plt)

    st.markdown("#### **Insights from EDA:**")
    st.write("""
    - The dataset contains material properties related to tensile strength.
    - Key relationships exist between strain, ultimate stress, and fiber properties.
    """)

# ✅ Curve Fitting
elif page == "Curve Fitting":
    st.title("📈 Curve Fitting for Stress-Strain Prediction")

    # Define 4th-degree polynomial model
    def polynomial_4th(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e

    X = df["Max_Stroke Strain Calc. at Entire Areas(% )"].values
    Y = df["Ultimate stress in N/mm^2"].values
    popt_poly4, _ = curve_fit(polynomial_4th, X, Y, maxfev=10000)

    # Plot curve fitting
    X_fit = np.linspace(min(X), max(X), 100)
    Y_fit_poly4 = polynomial_4th(X_fit, *popt_poly4)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X, y=Y, label="Actual Data", color="blue")
    plt.plot(X_fit, Y_fit_poly4, label="4th-Degree Polynomial Fit", linestyle="--", color="red")
    plt.xlabel("Max Stroke Strain (%)")
    plt.ylabel("Ultimate Stress (N/mm²)")
    plt.title("Polynomial Curve Fit")
    plt.legend()
    st.pyplot(plt)

    st.markdown("#### **Curve Fitting Insights:**")
    st.write("""
    - The 4th-degree polynomial model best represents the stress-strain behavior.
    - This model is used for reverse curve fitting to find strain at target stress levels.
    """)

# ✅ Artificial Neural Network (ANN) Model
elif page == "ANN Model":
    st.title("🤖 ANN Model for Tensile Strength Prediction")

    # Select input & output features
    X = df[[
        "Max_Stroke Strain Calc. at Entire Areas(% )",
        "No.of yarn",
        "Ultimate Load in KN",
        "Max_Force calc. at Entire Areas(kgf)"
    ]].values

    Y = df[["Ultimate stress in N/mm^2"]].values

    # Normalize Data
    scaler_X, scaler_Y = StandardScaler(), StandardScaler()
    X_scaled, Y_scaled = scaler_X.fit_transform(X), scaler_Y.fit_transform(Y)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # Build ANN Model
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="linear")
    ])

    # Compile Model
    model.compile(optimizer="adam", loss="mse")

    # Train Model
    history = model.fit(X_train, Y_train, epochs=500, batch_size=8, validation_data=(X_test, Y_test), verbose=0)

    # Predict & Evaluate
    Y_pred_scaled = model.predict(X_test)
    Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

    r2_ann = r2_score(Y_test, Y_pred)
    rmse_ann = np.sqrt(mean_squared_error(Y_test, Y_pred))

    st.write(f"🔹 **ANN Model Performance:**")
    st.write(f"   - **R² Score:** {r2_ann:.3f}")
    st.write(f"   - **RMSE:** {rmse_ann:.3f} N/mm²")

    # Scatter Plot: ANN Predictions vs Actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Y_test.flatten(), y=Y_pred.flatten(), color="blue", label="Predicted vs Actual")
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle="--", color="red", label="Ideal Line")
    plt.xlabel("Actual Stress (N/mm²)")
    plt.ylabel("Predicted Stress (N/mm²)")
    plt.title("ANN Predictions vs Actual")
    plt.legend()
    st.pyplot(plt)

    st.markdown("#### **ANN Model Insights:**")
    st.write("""
    - The ANN model predicts ultimate stress using strain, yarn count, and force properties.
    - A higher R² score means better generalization.
    """)

# ✅ Code Explorer
elif page == "Code Explorer":
    st.title("💻 Full Code for Each Step")

    options = {
        "EDA Code": "Exploratory Data Analysis (EDA)",
        "Curve Fitting Code": "Curve Fitting Model",
        "ANN Code": "Artificial Neural Network Model"
    }
    
    choice = st.selectbox("Select a section to view code", list(options.keys()))
    
    if choice == "EDA Code":
        st.code(open("eda.py").read(), language="python")
    elif choice == "Curve Fitting Code":
        st.code(open("curve_fitting.py").read(), language="python")
    elif choice == "ANN Code":
        st.code(open("ann_model.py").read(), language="python")
