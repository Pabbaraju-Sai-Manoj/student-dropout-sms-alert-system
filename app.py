import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
twilio_sid = os.getenv("TWILIO_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_sms_from = os.getenv("TWILIO_SMS_FROM")

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("📚 ShikshaLens – Dropout Risk Dashboard")

# Format phone numbers to Twilio standard
def format_phone(number):
    number = str(number).strip()
    if number.startswith("91") and not number.startswith("+"):
        return f"+{number}"
    elif number.startswith("+"):
        return number
    return None

# Upload student data
uploaded_file = st.file_uploader("Upload Student Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Extract features for prediction
    X = df[["attendance", "grade", "parent_income"]]

    # Predict dropout risk
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] * 100

    # Add predictions to DataFrame
    df["Dropout Risk (%)"] = probabilities.round(2)
    df["Dropout Risk"] = ["Yes" if p == 1 else "No" for p in predictions]

    # Reorder columns (include optional ones)
    column_order = ["attendance", "grade", "parent_income", "Dropout Risk (%)", "Dropout Risk"]
    if "guardian_phone" in df.columns:
        column_order.insert(0, "guardian_phone")
    if "student_name" in df.columns:
        column_order.insert(0, "student_name")

    df = df[[col for col in column_order if col in df.columns]]

    # Display results
    st.dataframe(df)

    # Summary
    dropout_count = df[df["Dropout Risk"] == "Yes"].shape[0]
    not_at_risk_count = df[df["Dropout Risk"] == "No"].shape[0]

    st.markdown("### 🧮 Summary")
    st.markdown(f"- **Total Students at Risk of Dropout**: `{dropout_count}`")
    st.markdown(f"- **Total Students Not at Risk**: `{not_at_risk_count}`")

    # SHAP Explanation
    st.subheader("🔍 SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.initjs()
    shap.summary_plot(shap_values, X, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

    # CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name='student_dropout_predictions.csv',
        mime='text/csv',
    )

    # Dropout Risk Scatter Plot
    st.subheader("📈 Dropout Risk Scatter Plot")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["attendance"], df["Dropout Risk (%)"], c=probabilities, cmap='coolwarm', edgecolors='k')
    ax.set_xlabel("Attendance")
    ax.set_ylabel("Dropout Risk (%)")
    ax.set_title("Dropout Risk vs Attendance")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Risk Intensity")
    st.pyplot(fig)

    # SMS Notification
    st.subheader("📲 Notifying Guardians via SMS")

    if not twilio_sid or not twilio_auth_token or not twilio_sms_from:
        st.warning("⚠️ Twilio credentials not configured properly in `.env`. SMS alerts not sent.")
    elif "guardian_phone" not in df.columns:
        st.warning("⚠️ 'guardian_phone' column not found in uploaded data. SMS alerts not sent.")
    else:
        try:
            client = Client(twilio_sid, twilio_auth_token)
            at_risk_df = df[df["Dropout Risk"] == "Yes"]
            sent_count = 0

            for i, row in at_risk_df.iterrows():
                guardian_number = format_phone(row["guardian_phone"])
                if not guardian_number:
                    st.warning(f"⚠️ Invalid phone number skipped: {row['guardian_phone']}")
                    continue

                if not (guardian_number.startswith("+") and len(guardian_number) > 10):
                    st.warning(f"⚠️ Invalid phone number skipped: {guardian_number}")
                    continue

                student_info = row.get("student_name", f"Student {i+1}")
                message_body = (
                    f"🚨 Alert: Your ward {student_info} is at HIGH RISK of dropping out.\n"
                    f"Attendance: {row['attendance']}%, Grade: {row['grade']}, "
                    f"Income: {row['parent_income']}\n"
                    f"Risk Score: {row['Dropout Risk (%)']}%"
                )

                try:
                    client.messages.create(
                        body=message_body,
                        from_=twilio_sms_from,
                        to=guardian_number
                    )
                    sent_count += 1
                except Exception as sms_err:
                    st.error(f"❌ SMS failed for {guardian_number}: {sms_err}")

            st.success(f"✅ SMS alerts sent for {sent_count} student(s) at risk.")

        except Exception as e:
            st.error(f"❌ Failed to initialize Twilio client: {e}")
