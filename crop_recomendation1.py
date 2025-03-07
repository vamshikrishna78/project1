import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit Title
st.title("Crop Recommendation System")

# Navigation
menu = ["Home", "Climate-Based Recommendation", "Weather Report"]
choice = st.sidebar.selectbox("Navigation", menu)

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("crop_data.csv")
    return data

df = load_data()

# Display Dataset
if choice == "Home":
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
    recommended_crop = prediction[0]
    st.write(f"The recommended crop is: **{recommended_crop}**")

    # NPK Ratio Pie Chart (Dynamically updated based on recommended crop)
    st.subheader(f"NPK Ratio for {recommended_crop}")
    crop_data = df[df["crop"] == recommended_crop][["N", "P", "K"]].mean()
    fig, ax = plt.subplots()
    ax.pie(crop_data, labels=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"NPK Ratio for {recommended_crop}")
    st.pyplot(fig)

elif choice == "Weather Report":
    # Weather Report Section
    st.subheader("Weather Report Based on District and Mandal")

    # Load the weather data
    @st.cache_data
    def load_weather_data(district, mandal):
        weather_data = pd.read_csv("weather.csv")
        filtered_data = weather_data[(weather_data['district'] == district) & (weather_data['mandal'] == mandal)]
        return filtered_data

    district = st.text_input("Enter District:")
    mandal = st.text_input("Enter Mandal:")

    if district and mandal:
        weather_df = load_weather_data(district, mandal)
        
        # Radio button for showing the weather dataset
        show_dataset = st.radio("Do you want to see the weather dataset?", ("No", "Yes"))
        
        if show_dataset == "Yes":
            st.subheader(f"Weather Data for {district}, {mandal}")
            st.dataframe(weather_df)

        if not weather_df.empty:
            # Displaying only Max and Min values
            st.write(f"Max and Min Temperature, Humidity, and Rainfall for {district}, {mandal}:")

            max_temp = weather_df["temp_max"].max()
            min_temp = weather_df["temp_min"].min()
            max_humidity = weather_df["humidity_max"].max()
            min_humidity = weather_df["humidity_min"].min()
            max_rainfall = weather_df["rainfall"].max()

            st.write(f"Max Temperature: {max_temp}°C")
            st.write(f"Min Temperature: {min_temp}°C")
            st.write(f"Max Humidity: {max_humidity}%")
            st.write(f"Min Humidity: {min_humidity}%")
            st.write(f"Max Rainfall: {max_rainfall} mm")

            # Bar Graph: Max and Min Temperature
            st.subheader("Max and Min Temperature")
            temp_data = {
                "Max Temperature": max_temp,
                "Min Temperature": min_temp
            }
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(temp_data.keys(), temp_data.values(), color=['orange', 'blue'])
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Max and Min Temperature")
            st.pyplot(fig)

            # Bar Graph: Max and Min Humidity
            st.subheader("Max and Min Humidity")
            humidity_data = {
                "Max Humidity": max_humidity,
                "Min Humidity": min_humidity
            }
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(humidity_data.keys(), humidity_data.values(), color=['green', 'lightblue'])
            ax.set_ylabel("Humidity (%)")
            ax.set_title("Max and Min Humidity")
            st.pyplot(fig)

            # Bar Graph: Max Rainfall
            st.subheader("Max Rainfall")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(["Max Rainfall"], [max_rainfall], color='blue')
            ax.set_ylabel("Rainfall (mm)")
            ax.set_title("Max Rainfall")
            st.pyplot(fig)

        else:
            st.write("No weather data found for the given district and mandal.")
    else:
        st.write("Please enter both district and mandal.")

elif choice == "Climate-Based Recommendation":
    st.subheader("Climate-Based Crop Recommendation")

    # Get user climate input
    st.sidebar.header("Input Climate Conditions")
    user_temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    user_humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    user_rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 400.0, 50.0)

    # Calculate the similarity between user's climate and crop's required climate
    df["climate_similarity"] = np.sqrt(
        (df["temperature"] - user_temp)**2 +
        (df["humidity"] - user_humidity)**2 +
        (df["rainfall"] - user_rainfall)**2
    )

    # Find the crop with the least climate similarity
    recommended_crop_climate = df.loc[df["climate_similarity"].idxmin()]["crop"]
    st.write(f"The recommended crop based on climate conditions is: **{recommended_crop_climate}**")
