import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import streamlit as st
import pickle
import plotly.graph_objects as go


def preprocess_data(df):
    """Preprocesses the data: label encoding."""
    label_encoder = LabelEncoder()
    df['Project_complete'] = label_encoder.fit_transform(df['Project_complete'])
    return df

def remove_outliers_from_data(df):
    """
    Removes outliers from the data using DBSCAN clustering.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        cleaned_data (pd.DataFrame): The dataframe with outliers removed.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    dbscan = DBSCAN(eps=7.3, min_samples=5)
    dbscan.fit(scaled_data)

    df['Cluster'] = dbscan.labels_
    cleaned_data = df[df['Cluster'] != -1]  # Remove outliers (label = -1)

    return cleaned_data.drop(columns=['Cluster'], errors='ignore')

def normalize_data(df):
    """
    Normalizes the dataframe using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        df_normalized (pd.DataFrame): The normalized dataframe.
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def train_and_evaluate_logistic_regression(x_train, y_train, x_test, y_test):
    """
    Trains a logistic regression model using SMOTE-augmented data.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
    """

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    roc_auc = evaluate_model(model, x_test, y_test)
    print(f"Logistic Regression ROC AUC: {roc_auc:.4f}")
    return model

def train_and_evaluate_decision_tree(x_train, y_train, x_test, y_test, cv=5):
    """Trains and evaluates Decision Tree with cross-validation and GridSearch."""
    param_grid = {'max_depth': range(1, 11), 'min_samples_split': range(2, 11)}
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(x_train, y_train)
    print(f"Decision Tree Best Parameters: {grid_search.best_params_}")
    print(f"Decision Tree CV ROC AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate_random_forest(x_train, y_train, x_test, y_test, cv=5):
    """Trains and evaluates Random Forest with cross-validation and GridSearch."""
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"Random Forest Best Parameters: {grid_search.best_params_}")
    print(f"Random Forest CV ROC AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate_gradient_boosting(x_train, y_train, x_test, y_test, cv=5):
    """Trains and evaluates Gradient Boosting with cross-validation and GridSearch."""
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=cv, scoring='roc_auc')
    grid_search.fit(x_train, y_train)
    print(f"Gradient Boosting Best Parameters: {grid_search.best_params_}")
    print(f"Gradient Boosting CV ROC AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate_svm(x_train, y_train, x_test, y_test, cv=5):
    """Trains and evaluates SVM with cross-validation and GridSearch."""
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'kernel': ['rbf']}
    grid_search = GridSearchCV(SVC(probability=True, random_state=42), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)  # probability=True is important for ROC AUC
    grid_search.fit(x_train, y_train)
    print(f"SVM Best Parameters: {grid_search.best_params_}")
    print(f"SVM CV ROC AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate_knn(x_train, y_train, x_test, y_test, cv=5):
    """Trains and evaluates KNN with cross-validation and GridSearch."""
    param_grid = {'n_neighbors': range(1, 21)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"KNN Best Parameters: {grid_search.best_params_}")
    print(f"KNN CV ROC AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def train_and_evaluate_ann(x_train, y_train, x_test, y_test, epochs=100, batch_size=32):
    """
    Trains and evaluates an Artificial Neural Network.

    Args:
        x_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        x_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        model: Trained ANN model.
    """
    # Define the ANN structure
    model = Sequential([
        Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    y_pred_proba = model.predict(x_test).flatten()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ANN ROC AUC: {roc_auc:.4f}")

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained logistic regression model using ROC AUC score.

    Args:
        model: Trained LogisticRegression model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing target.

    Returns:
        roc_auc (float): ROC AUC score.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

def plot_roc_curve(model, X_test, y_test, model_name):
    """Plots the ROC curve."""
    if hasattr(model, "predict_proba"):  # For sklearn models
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # For ANN or other models with predict() method
        y_pred_proba = model.predict(X_test).flatten()

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

def train_models(df):
    """Trains all models and returns them in a dictionary."""
    df_processed = preprocess_data(df.copy())
    df_cleaned = remove_outliers_from_data(df_processed)
    df_normalized = normalize_data(df_cleaned)

    x = df_normalized.drop('Project_complete', axis=1)
    y = df_normalized['Project_complete']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    models = {
        "Logistic Regression": train_and_evaluate_logistic_regression(x_train_smote, y_train_smote, x_test, y_test),
        "Decision Tree": train_and_evaluate_decision_tree(x_train_smote, y_train_smote, x_test, y_test),
        "Random Forest": train_and_evaluate_random_forest(x_train_smote, y_train_smote, x_test, y_test),
        "Gradient Boosting": train_and_evaluate_gradient_boosting(x_train_smote, y_train_smote, x_test, y_test),
        "SVM": train_and_evaluate_svm(x_train_smote, y_train_smote, x_test, y_test),
        "KNN": train_and_evaluate_knn(x_train_smote, y_train_smote, x_test, y_test),
        "ANN": train_and_evaluate_ann(x_train_smote, y_train_smote, x_test, y_test)
    }
    return models, x_test, y_test, x_train.columns  # return columns for input


def make_recommendations(prediction, model_name, student_data, feature_means):
    """Provides detailed recommendations with charts and comparisons."""
    student_name = student_data.get('student_name', "Unknown Student")
    student_id = student_data.get('student_id', "N/A")

    with st.expander(f"**Recommendations for: {student_name} (ID: {student_id})**"):  # Collapsible section
        if prediction == 1:
            st.success(f"Predicted to complete the project (using {model_name}).")
            st.write("Maintaining Performance:")
            st.write("- Continue current study habits. Your current metrics suggest good engagement.")
        else:
            st.warning(f"Predicted to NOT complete the project (using {model_name}).")
            st.write("Areas for Improvement:")

            features = []
            student_values = []
            average_values = []
            comparison_texts = []

            for feature, value in student_data.items():
                if feature not in ['student_name', 'student_id']:
                    mean_val = feature_means.get(feature)
                    if mean_val is not None:
                        difference = value - mean_val
                        comparison_text = "above average" if difference > 0 else "below average"
                        features.append(feature)
                        student_values.append(value)
                        average_values.append(mean_val)
                        comparison_texts.append(comparison_text)

            # Bar Chart Comparison using Plotly
            fig = go.Figure(data=[
                go.Bar(name='Student Value', x=features, y=student_values),
                go.Bar(name='Average Value', x=features, y=average_values)
            ])
            fig.update_layout(barmode='group', title="Feature Comparison", yaxis_title="Value")
            st.plotly_chart(fig)

            for i, feature in enumerate(features):
                st.write(f"- Your **{feature}** ({student_values[i]:.2f}) is {comparison_texts[i]} compared to the average ({average_values[i]:.2f}).")
                if comparison_texts[i] == "below average":
                    st.write(f"  * Consider focusing on improving {feature} to enhance your project completion chances.")

            st.write("Additional Suggestions:")
            st.write("- Seek extra help from instructors or tutors.")
            st.write("- Review course materials regularly.")
            st.write("- Form study groups with classmates.")
            st.write("- Improve time management skills.")

        st.write("---")

def analyze_results(predictions, student_data_list, feature_means):
    """Analyzes the results and provides a summary for managers."""
    num_students = len(predictions)
    good_results = sum(predictions)
    bad_results = num_students - good_results

    st.subheader("Overall Summary for Managers")
    st.write(f"Total Students: {num_students}")
    st.write(f"Predicted to Complete Project: {good_results} ({good_results/num_students*100:.2f}%)")
    st.write(f"Predicted to NOT Complete Project: {bad_results} ({bad_results/num_students*100:.2f}%)")

    # Worst and Best Performing Students (based on number of below-average features)
    performance = []
    for i, student_data in enumerate(student_data_list):
        below_avg_count = 0
        for feature, value in student_data.items():
            if feature not in ['student_name', 'student_id']:
                mean_val = feature_means.get(feature)
                if mean_val is not None and value < mean_val:
                    below_avg_count += 1
        performance.append((below_avg_count, student_data))

    performance.sort(key=lambda x: x[0])  # Sort by number of below-average features

    if performance:
        worst_performing = performance[-1][1]
        best_performing = performance[0][1]

        st.write(f"Worst Performing Student: {worst_performing.get('student_name', 'N/A')} (ID: {worst_performing.get('student_id', 'N/A')}, {performance[-1][0]} below average features)")
        st.write(f"Best Performing Student: {best_performing.get('student_name', 'N/A')} (ID: {best_performing.get('student_id', 'N/A')}, {performance[0][0]} below average features)")
    else:
        st.write("No student data to analyze.")

    # Manager Recommendations
    st.subheader("Manager Recommendations")
    if bad_results > num_students * 0.3: # If more than 30% are predicted to fail
        st.write("- Implement targeted interventions for at-risk students, focusing on the features identified as below average.")
        st.write("- Review course materials and teaching methods to ensure they are engaging and effective.")
        st.write("- Consider providing additional resources such as tutoring or study groups.")
    elif bad_results > 0:
        st.write("- Monitor the performance of students predicted to not complete the project and provide support as needed.")
        st.write("- Analyze the features contributing to poor performance to identify areas for improvement in the curriculum or support services.")
    else:
        st.write("- The overall performance is very good. Continue current strategies and monitor for any changes.")
    st.write("- Track student performance over time to assess the effectiveness of interventions and strategies.")

def main():
    st.title("Student Project Completion Prediction")

    uploaded_file = st.file_uploader("Upload CSV (Training Data)", type=["csv"])
    if uploaded_file is not None:
        try:
            train_df = pd.read_csv(uploaded_file)
            models, x_test, y_test, input_cols = train_models(train_df)
            st.write("Models trained successfully!")

            feature_means = train_df[input_cols].mean()  # calculate means for comparison

            new_data_file = st.file_uploader("Upload CSV (New Student Data)", type=["csv"])
            if new_data_file is not None:
                new_df = pd.read_csv(new_data_file)

                if 'Project_complete' in new_df.columns:
                    new_df = new_df.drop(columns=['Project_complete'])  # drop if it exists

                student_data_list = new_df.to_dict(orient='records')  # list of dictionaries
                ids_names = new_df[['student_id',
                                    'student_name']] if 'student_id' in new_df.columns and 'student_name' in new_df.columns else None

                if 'student_id' in new_df.columns:
                    new_df = new_df.drop(columns=['student_id'])
                if 'student_name' in new_df.columns:
                    new_df = new_df.drop(columns=['student_name'])

                try:
                    new_df = normalize_data(new_df)
                    selected_model = st.selectbox("Select a model", list(models.keys()))
                    model = models[selected_model]
                    predictions = model.predict(new_df)
                    all_predictions = []

                    for i, prediction in enumerate(predictions):
                        student_data = student_data_list[i]
                        make_recommendations(prediction, selected_model, student_data, feature_means)
                        all_predictions.append(prediction)

                    plot_roc_curve(model, x_test, y_test, selected_model)
                    st.pyplot()

                    analyze_results(np.array(all_predictions), student_data_list, feature_means)  # analyze results

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.info("Please upload new student data for predictions.")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    main()