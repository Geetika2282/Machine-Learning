import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Sample data for training the model
data = {
    'courage': [9, 7, 8, 5, 7],
    'intelligence': [7, 10, 7, 8, 8],
    'loyalty': [10, 9, 8, 4, 9],
    'ambition': [5, 6, 5, 9, 4],
    'kindness': [8, 9, 8, 3, 10],
    'character': ['Harry Potter', 'Hermione Granger', 'Ron Weasley', 'Draco Malfoy', 'Luna Lovegood']
}

df = pd.DataFrame(data)

# Step 2: Preparing the model
X = df[['courage', 'intelligence', 'loyalty', 'ambition', 'kindness']]
y = df['character']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

# Streamlit App
st.title("Which Harry Potter Character Are You?")

# Input fields for user traits
courage = st.number_input("Rate your courage (0 to 10):", 0.0, 10.0, step=0.1)
intelligence = st.number_input("Rate your intelligence (0 to 10):", 0.0, 10.0, step=0.1)
loyalty = st.number_input("Rate your loyalty (0 to 10):", 0.0, 10.0, step=0.1)
ambition = st.number_input("Rate your ambition (0 to 10):", 0.0, 10.0, step=0.1)
kindness = st.number_input("Rate your kindness (0 to 10):", 0.0, 10.0, step=0.1)

# Prediction button
if st.button("Find Your Character"):
    user_data = pd.DataFrame({
        'courage': [courage],
        'intelligence': [intelligence],
        'loyalty': [loyalty],
        'ambition': [ambition],
        'kindness': [kindness]
    })

    user_data_scaled = scaler.transform(user_data)
    predicted_character = model.predict(user_data_scaled)[0]

    # Display the result
    st.success(f"You resemble: {predicted_character}")

