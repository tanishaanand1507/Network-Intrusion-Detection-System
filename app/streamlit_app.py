import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# PAGE CONFIG
st.set_page_config(
    page_title="Sentinel AI",
    page_icon="🛡️",
    layout="wide"
)

# CUSTOM STYLING
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #050816, #12051a, #07121f);
    color: white;
}

h1 {
    color: #00FFAA;
    text-align: center;
    font-size: 52px;
    text-shadow: 0px 0px 15px #00FFAA;
}

h2, h3 {
    color: #B026FF;
}

section[data-testid="stSidebar"] {
    background-color: #0D1117;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(176, 38, 255, 0.4);
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL
model = joblib.load('models/intrusion_model.pkl')

# TITLE
st.title("🛡️ Sentinel AI")

st.markdown("""
<h3 style='text-align:center;color:white;'>
AI-Powered Intrusion Detection Dashboard
</h3>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("⚡ Sentinel AI")

st.sidebar.info("""
Machine Learning Based Cybersecurity Dashboard
""")

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# IF FILE EXISTS
if uploaded_file is not None:

    # READ CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data")
    st.dataframe(data.head())

    # ENCODE CATEGORICAL COLUMNS
    categorical_columns = [
        'protocol_type',
        'service',
        'flag'
    ]

    encoder = LabelEncoder()

    for col in categorical_columns:

        if col in data.columns:
            data[col] = encoder.fit_transform(data[col])

    # MAKE PREDICTIONS
    prediction = model.predict(data)

    # ADD PREDICTIONS
    data['Prediction'] = prediction

    # COUNT RESULTS
    attack_count = int(sum(prediction))
    normal_count = int(len(prediction) - attack_count)

    # THREAT ALERTS
    if attack_count > 50:

        st.error(
            "🚨 CRITICAL NETWORK THREAT DETECTED"
        )

    elif attack_count > 20:

        st.warning(
            "⚠ HIGH THREAT ACTIVITY DETECTED"
        )

    else:

        st.success(
            "✅ Network Traffic Appears Safe"
        )

    # METRICS
    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class='metric-card'>
        <h2 style='color:#00FFAA;'>🟢 Normal</h2>
        <h1>{normal_count}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class='metric-card'>
        <h2 style='color:#FF4B4B;'>🚨 Attacks</h2>
        <h1>{attack_count}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        total = normal_count + attack_count

        threat_percentage = round(
            (attack_count / total) * 100,
            2
        )

        st.markdown(f"""
        <div class='metric-card'>
        <h2 style='color:#B026FF;'>⚠ Threat Level</h2>
        <h1>{threat_percentage}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # LIVE SOC METRICS
    st.subheader("📡 Live SOC Metrics")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Total Packets",
        len(data)
    )

    metric2.metric(
        "Threat %",
        f"{threat_percentage}%"
    )

    health_score = 100 - threat_percentage

    metric3.metric(
        "Traffic Health",
        f"{health_score}%"
    )

    st.markdown("---")

    # RESULTS
    st.subheader("🔍 Prediction Results")

    st.dataframe(data.head(20))

    # PIE CHART
    st.subheader("📊 Threat Distribution")

    labels = ['Normal', 'Attack']
    values = [normal_count, attack_count]

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    st.pyplot(fig)

    # DOWNLOAD BUTTON
    csv = data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇ Download Results",
        data=csv,
        file_name='prediction_results.csv',
        mime='text/csv'
    )

# NO FILE
else:

    st.markdown("""
    <div style='text-align:center;padding:50px;'>
    <h2 style='color:#00FFAA;'>
    Upload sample_test.csv To Start Analysis
    </h2>
    </div>
    """, unsafe_allow_html=True)

