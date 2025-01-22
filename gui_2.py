import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Import functions from your existing script
from main import (
    preprocess_data, remove_outliers_from_data, normalize_data,
    train_and_evaluate_logistic_regression, train_and_evaluate_decision_tree,
    train_and_evaluate_random_forest, train_and_evaluate_gradient_boosting,
    train_and_evaluate_svm, train_and_evaluate_knn, train_and_evaluate_ann,
    plot_roc_curve
)

def full_workflow(df):
    """Executes all steps of the workflow."""
    st.write("Starting Preprocessing...")
    try:
        # Preprocess
        df = preprocess_data(df)
        st.success("Data Preprocessed Successfully!")

        # Remove Outliers
        df = remove_outliers_from_data(df)
        st.success("Outliers Removed Successfully!")

        # Normalize Data
        df = normalize_data(df)
        st.success("Data Normalized Successfully!")

        # Split Data
        x = df.drop("Project_complete", axis=1)
        y = df["Project_complete"]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

        # Handle Imbalanced Data with SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

        st.write("Starting Model Training...")
        trained_models = {}

        # Train Models
        trained_models["Logistic Regression"] = train_and_evaluate_logistic_regression(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["Decision Tree"] = train_and_evaluate_decision_tree(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["Random Forest"] = train_and_evaluate_random_forest(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["Gradient Boosting"] = train_and_evaluate_gradient_boosting(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["SVM"] = train_and_evaluate_svm(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["KNN"] = train_and_evaluate_knn(
            x_train_smote, y_train_smote, x_test, y_test
        )
        trained_models["ANN"] = train_and_evaluate_ann(
            x_train_smote, y_train_smote, x_test, y_test
        )

        st.success("Model Training Completed!")

        # Evaluate Models
        st.write("Starting Evaluation...")
        for model_name, model in trained_models.items():
            st.subheader(f"ROC Curve for {model_name}")
            st.write(plot_roc_curve(model, x_test, y_test, model_name))

        st.success("Workflow Completed!")
    except Exception as e:
        st.error(f"Error during workflow: {e}")

# Streamlit App
def main():
    st.title("Machine Learning Workflow GUI")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Full Workflow"])

    if page == "Home":
        st.header("Welcome to the ML Workflow App")
        st.write("Upload your dataset and start the machine learning workflow.")

    elif page == "Full Workflow":
        st.header("Run Full Workflow")

        # File Upload
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df)

            # Run Full Workflow
            if st.button("Run Full Workflow"):
                full_workflow(df)

if __name__ == "__main__":
    main()
