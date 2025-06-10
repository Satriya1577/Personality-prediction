import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Load dataset
df = pd.read_csv("personality_dataset.csv")

# Show the first few rows (optional)
print(df.head())

# Encode target column: introvert=0, extrovert=1
label_encoder = LabelEncoder()
df['Personality'] = label_encoder.fit_transform(df['Personality'])

# Replace 'yes' with 1 and 'no' with 0 in the 'Stage_fear' column
df["Stage_fear"] = df["Stage_fear"].replace({"Yes": 1, "No": 0, "null": -1})

# Replace 'yes' with 1 and 'no' with 0 in the 'Drained_after_socializing' column
df["Drained_after_socializing"] = df["Drained_after_socializing"].replace({"Yes": 1, "No": 0, "null": -1})

# Separate features and target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

st.title("🧠 Personality Prediction Form")

with st.form("personality_form"):
    st.header("Please fill in the details:")

    # Sliders (scale 0–10)
    time_spent_Alone = st.slider("Waktu yang dihabiskan untuk sendirian", 0, 11)
    social_event_attendance = st.slider("Frekuensi kehadiran acara", 0, 10)
    going_outside = st.slider("Frekuensi pergi keluar rumah", 0, 10)
    friends_circle_size = st.slider("Ukuran lingkaran pertemanan", 0, 15)
    post_frequency = st.slider("Frekuensi posting foto di medsos", 0, 10)
  

    # Yes/No options
    stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
    drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])

    # Submit button
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Convert yes/no to numeric
        stage_fear_val = 1 if stage_fear == "Yes" else 0
        drained_val = 1 if drained_after_socializing == "Yes" else 0

        # Show collected data (or pass to model)
        st.subheader("🔍 Collected Inputs:")
        st.write({
            "Time Spent Alone": time_spent_Alone,
            "Social Event Attendance": social_event_attendance,
            "Going Outside": going_outside,
            "Friends Circle": friends_circle_size,
            "Post Frequenct": post_frequency,
            "Stage Fear": stage_fear_val,
            "Drained After Socializing": drained_val
        })

        # Time_spent_Alone,Stage_fear,Social_event_attendance,Going_outside,Drained_after_socializing,Friends_circle_size,Post_frequency,Personality

        new_data = pd.DataFrame([[time_spent_Alone, 
                                  stage_fear_val, 
                                  social_event_attendance, 
                                  going_outside, 
                                  drained_val, 
                                  friends_circle_size, 
                                  post_frequency]], columns=X.columns)

        prediction = clf.predict(new_data)
        predicted_label = label_encoder.inverse_transform(prediction)
        st.success(f"🧠 Predicted Personality: {predicted_label[0]}")





