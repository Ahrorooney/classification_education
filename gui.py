import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import numpy as np

# Simulated dataset for demonstration
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Feature1': np.random.rand(100),
        'Feature2': np.random.rand(100),
        'Label': np.random.choice([0, 1], size=100)
    })
    return data

data = load_data()

# Train a sample Random Forest model
def train_model(data):
    X = data[['Feature1', 'Feature2']]
    y = data['Label']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model, X, y

model, X, y = train_model(data)

# Calculate ROC AUC
probs = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

# Streamlit App
st.title("AI Project GUI: Decision Layer")

# Sidebar for model selection
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose a Model",
    ["Random Forest"]  # Add more models as needed
)

st.sidebar.markdown(f"### Selected Model: {selected_model}")

# Main Content
st.header("Dataset Overview")
st.write("Sample of the dataset:")
st.dataframe(data.head())

# Visualize Feature Importance
st.header("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': ['Feature1', 'Feature2'],
    'Importance': model.feature_importances_
})
fig = px.bar(feature_importance, x='Feature', y='Importance', title="Feature Importance")
st.plotly_chart(fig)

# ROC Curve Visualization
st.header("Model Performance: ROC Curve")
fig_roc = px.area(
    x=fpr, y=tpr,
    title=f"ROC Curve (AUC = {roc_auc:.2f})",
    labels=dict(x="False Positive Rate", y="True Positive Rate"),
    width=700, height=500
)
fig_roc.add_shape(
    type="line", line=dict(dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)
st.plotly_chart(fig_roc)

# Recommendation Section
st.header("Generate Recommendations")
if st.button("Generate Recommendations"):
    st.success("Recommendations Generated Successfully!")
    st.write(
        "- **Group A:** Students requiring personalized learning paths.\n"
        "- **Group B:** Targeted campaigns for improved engagement.\n"
        "- **Group C:** High-performing students to track for scholarships."
    )

st.write("\n---\n")
st.info("Developed for AI project decision layer.")
