import streamlit as st
import pickle
import numpy as np


st.image("titanic.png", use_container_width=True, caption="Titanic - Predict Survival")


# Load models
logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
decision_tree_model = pickle.load(open("decision_tree_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details below to predict survival.")

# Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=500.0, value=50.0)

# Convert gender to numeric
sex = 1 if sex == "Male" else 0

# Add Embarked input
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode embarked (same as in model training)
if embarked == 'C':
    embarked = 0
elif embarked == 'Q':
    embarked = 1
else:
    embarked = 2

# Combine into array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Choose model
model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "SVM"])

if st.button("Predict Survival"):
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(features)
    elif model_choice == "Decision Tree":
        prediction = decision_tree_model.predict(features)
    else:
        prediction = svm_model.predict(features)
    
    if prediction[0] == 1:
        st.success("🎉 The passenger would have **survived**!")
    else:
        st.error("❌ The passenger would **not have survived**.")
