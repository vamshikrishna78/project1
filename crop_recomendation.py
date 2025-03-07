import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Streamlit Title
st.title("Crop Recommendation System")

# Load Dataset
@st.cache_data
def load_data():
    # Replace 'crop_data.csv' with the actual dataset file path
    data = pd.read_csv("crop_data.csv")
    return data

df = load_data()

# Display Dataset
if st.checkbox("Show Dataset"):
    st.subheader("Dataset")
    st.dataframe(df)

# Split Dataset into Features (X) and Target (y)
X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["crop"]

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate Model Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the Model Accuracy
st.write(f"Model Accuracy: **{accuracy * 100:.2f}%**")

# User Input Section
st.sidebar.header("Input Parameters")
def user_input_features():
    N = st.sidebar.slider("Nitrogen (N)", 0, 300, 50)
    P = st.sidebar.slider("Phosphorus (P)", 0, 300, 50)
    K = st.sidebar.slider("Potassium (K)", 0, 300, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 400.0, 50.0)
    
    input_data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }
    return pd.DataFrame(input_data, index=[1])

input_df = user_input_features() 

# Predict Crop
st.subheader("Your Input Parameters")
st.write(input_df)

prediction = model.predict(input_df)
st.subheader("Recommended Crop")
st.write(f"The recommended crop is: **{prediction[0]}**")

# Instructions for Dataset
st.info("""
The dataset must include the following columns:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **temperature**: Temperature in degrees Celsius
- **humidity**: Relative humidity in percentage
- **ph**: Soil pH value
- **rainfall**: Rainfall in mm
- **crop**: Target crop name
""")

# NPK Analysis - Bar Graph
st.subheader("Average NPK Content for Each Crop")
average_npk = df.groupby("crop")[["N", "P", "K"]].mean()
st.bar_chart(average_npk)

# Dynamically updated NPK Ratio Pie Chart
st.subheader("NPK Ratio for Your Recommended Crop")
selected_crop = prediction[0]  # Using the predicted crop

if selected_crop:
    crop_data = df[df["crop"] == selected_crop][["N", "P", "K"]].mean()
    fig, ax = plt.subplots()
    ax.pie(crop_data, labels=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"NPK Ratio for {selected_crop}")
    st.pyplot(fig)

# NPK Requirement Analysis - Pie Chart
st.subheader("NPK Requirements for All Crops")
npk_sum = average_npk.sum()
fig, ax = plt.subplots()
ax.pie(npk_sum, labels=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"], autopct="%1.1f%%", startangle=90)
ax.set_title("NPK Requirements for All Crops")
st.pyplot(fig)

# pH Analysis for Each Crop
st.subheader("Soil pH Analysis for Each Crop")

# Classify crops as Acidic or Basic based on pH
df["pH_Status"] = df["ph"].apply(lambda x: "Acidic" if x < 7.0 else "Basic")

# Bar Graph: Count of Acidic vs. Basic Crops
st.subheader("Count of Acidic vs. Basic Crops")
ph_counts = df["pH_Status"].value_counts()
st.bar_chart(ph_counts)

# Pie Chart: Proportion of Acidic vs. Basic Crops
st.subheader("Proportion of Acidic vs. Basic Crops")
fig, ax = plt.subplots()
ax.pie(ph_counts, labels=ph_counts.index, autopct="%1.1f%%", startangle=90, colors=["#FF9999", "#66B3FF"])
ax.set_title("Acidic vs. Basic Crops")
st.pyplot(fig)

# Bar Graph: pH Value for Each Crop
st.subheader("pH Value for Each Crop")
average_ph = df.groupby("crop")["ph"].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(average_ph.index, average_ph.values, color=["#FF9999" if x < 7.0 else "#66B3FF" for x in average_ph.values])
ax.set_xlabel("Average pH")
ax.set_ylabel("Crops")
ax.set_title("pH Values for Crops (Acidic or Basic)")
ax.bar_label(bars, fmt="%.2f")
st.pyplot(fig)
print(pd.options.display.max_rows)
print('welcome to the croprecommendation')