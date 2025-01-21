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

# Streamlit App
def main():
    st.title("Machine Learning Workflow GUI")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Home", "Preprocessing", "Model Training", "Evaluation"])

    if page == "Home":
        st.header("Welcome to the ML Workflow App")
        st.write("Upload your dataset and start the machine learning workflow.")

    elif page == "Preprocessing":
        st.header("Data Preprocessing")

        # File Upload
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df)

            # Preprocessing options
            if st.button("Preprocess Data"):
                try:
                    df = preprocess_data(df)
                    st.success("Data Preprocessed Successfully!")
                    st.write(df.head())
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.button("Remove Outliers"):
                df = remove_outliers_from_data(df)
                st.success("Outliers Removed!")

            if st.button("Normalize Data"):
                df = normalize_data(df)
                st.success("Data Normalized!")

    elif page == "Model Training":
        st.header("Train Machine Learning Models")

        # Load dataset
        uploaded_file = st.file_uploader("Upload CSV for Model Training", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df = preprocess_data(df)
            df = remove_outliers_from_data(df)
            df = normalize_data(df)

            x = df.drop("Project_complete", axis=1)
            y = df["Project_complete"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

            smote = SMOTE(random_state=42, k_neighbors=3)
            x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

            model_choice = st.multiselect("Select Models to Train:", ["Logistic Regression", "Decision Tree",
                                                                      "Random Forest", "Gradient Boosting",
                                                                      "SVM", "KNN", "ANN"])
            trained_models = {}

            if "Logistic Regression" in model_choice:
                lr_model = train_and_evaluate_logistic_regression(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["Logistic Regression"] = lr_model

            if "Decision Tree" in model_choice:
                dt_model = train_and_evaluate_decision_tree(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["Decision Tree"] = dt_model

            if "Random Forest" in model_choice:
                rf_model = train_and_evaluate_random_forest(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["Random Forest"] = rf_model

            if "Gradient Boosting" in model_choice:
                gb_model = train_and_evaluate_gradient_boosting(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["Gradient Boosting"] = gb_model

            if "SVM" in model_choice:
                svm_model = train_and_evaluate_svm(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["SVM"] = svm_model

            if "KNN" in model_choice:
                knn_model = train_and_evaluate_knn(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["KNN"] = knn_model

            if "ANN" in model_choice:
                ann_model = train_and_evaluate_ann(x_train_smote, y_train_smote, x_test, y_test)
                trained_models["ANN"] = ann_model

            st.write("Model Training Completed!")

    elif page == "Evaluation":
        st.header("Model Evaluation")

        for model_name, model in trained_models.items():
            st.subheader(f"ROC Curve for {model_name}")
            plot_roc_curve(model, x_test, y_test, model_name)

if __name__ == "__main__":
    main()
