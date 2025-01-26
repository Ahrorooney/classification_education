import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score, roc_curve
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

    return model

def train_models(df, show_process=False):
    """Trains models and displays processing steps with ROC curves."""
    original_df = df.copy()
    if show_process:
        st.subheader("Data Before Preprocessing")
        st.write(original_df.head())
        st.write(f"Shape: {original_df.shape}")

    df_processed = preprocess_data(df.copy())
    if show_process:
        st.subheader("Data After Preprocessing (Label Encoding)")
        st.write(df_processed.head())

    df_cleaned = remove_outliers_from_data(df_processed)
    if show_process:
        st.subheader("Data After Outlier Removal (DBSCAN)")
        st.write(df_cleaned.head())
        st.write(f"Shape: {df_cleaned.shape} (Outliers removed)")

    df_normalized = normalize_data(df_cleaned)
    if show_process:
        st.subheader("Data After Normalization (Min-Max Scaling)")
        st.write(df_normalized.head())

    x = df_normalized.drop('Project_complete', axis=1)
    y = df_normalized['Project_complete']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=min(5, len(x_train_scaled) - 1))
    x_train_smote, y_train_smote = smote.fit_resample(x_train_scaled, y_train)

    if show_process:
        st.subheader("Training Data After SMOTE")
        st.write(pd.DataFrame(x_train_smote).head())
        st.write(f"Shape of X_train after SMOTE: {x_train_smote.shape}")
        st.write(f"Shape of y_train after SMOTE: {y_train_smote.shape}")

    # Scoring dictionary for GridSearchCV
    scoring = {'roc_auc': 'roc_auc',
               'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1': make_scorer(f1_score)}

    # Define models HERE, before the if show_process block
    models = {
        "Logistic Regression": GridSearchCV(LogisticRegression(max_iter=5000, penalty='l2', C=10, solver='liblinear'),
                                            param_grid={}, scoring=scoring, refit='roc_auc', cv=cv,
                                            return_train_score=True, n_jobs=-1).fit(x_train_smote,
                                                                                    y_train_smote).best_estimator_,
        # Increased C
        "Decision Tree": GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'),
                                      {'max_depth': [2, 3], 'min_samples_split': [10, 20], 'min_samples_leaf': [5, 10]},
                                      scoring=scoring, refit='roc_auc', cv=cv, return_train_score=True, n_jobs=-1).fit(
            x_train_smote, y_train_smote).best_estimator_,  # More aggressive pruning
        "Random Forest": GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                                      {'n_estimators': [30, 50], 'max_depth': [3, 5], 'max_features': ['sqrt']},
                                      scoring=scoring, refit='roc_auc', cv=cv, return_train_score=True, n_jobs=-1).fit(
            x_train_smote, y_train_smote).best_estimator_,  # More aggressive pruning
        "Gradient Boosting": GridSearchCV(GradientBoostingClassifier(random_state=42),
                                          {'n_estimators': [30, 50], 'learning_rate': [0.01, 0.1], 'max_depth': [2, 3]},
                                          scoring=scoring, refit='roc_auc', cv=cv, return_train_score=True).fit(
            x_train_smote, y_train_smote).best_estimator_,  # More aggressive pruning
        "SVM": GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'),
                            {'C': [1, 10], 'gamma': [0.001, 0.01], 'kernel': ['rbf']}, scoring=scoring, refit='roc_auc',
                            cv=cv, return_train_score=True, n_jobs=-1).fit(x_train_smote,
                                                                           y_train_smote).best_estimator_,
        # Increased C, decreased gamma
        "KNN": GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [15, 20]}, scoring=scoring, refit='roc_auc', cv=cv,
                            return_train_score=True, n_jobs=-1).fit(x_train_smote, y_train_smote).best_estimator_,
        # Increased n_neighbors
        "ANN": train_and_evaluate_ann(x_train_smote, y_train_smote, x_test_scaled, y_test)
    }

    if show_process:
        st.subheader("Model Evaluation on Test Data")
        for model_name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(x_test)[:, 1]
            else:  # For ANN and other models without predict_proba
                y_pred_proba = model.predict(x_test).flatten()  # Flatten the predictions

            # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
            y_pred = (y_pred_proba > 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"**{model_name}**")
            st.write(f"ROC AUC: {roc_auc:.4f}")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1-Score: {f1:.4f}")

            # Handle the case for ANN (no classes_ attribute)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
            disp.plot()
            st.pyplot(plt)
            plt.clf()

            # ROC Curve using Plotly
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fig = go.Figure(data=[go.Scatter(x=fpr, y=tpr, name=f"{model_name} (area = {roc_auc:.2f})")])
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
            fig.update_layout(title=f'ROC Curve - {model_name}', xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate', width=400, height=400)
            st.plotly_chart(fig)

    return models, x_test, y_test, x_train.columns, original_df

def make_recommendations(prediction, model_name, student_data, feature_means):
    """Provides recommendations with a summary and detailed breakdown."""
    student_name = student_data.get('student_name', "Unknown Student")
    student_id = student_data.get('student_id', "N/A")

    with st.expander(f"**Recommendations for: {student_name} (ID: {student_id})**"):
        if prediction == 1:
            st.success(f"Predicted to complete the project (using {model_name}).")
            st.write("Overall, your metrics suggest good engagement. Continue your current study habits.")
        else:
            st.warning(f"Predicted to NOT complete the project (using {model_name}).")

            below_avg_features = []
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
                        if comparison_text == "below average":
                            below_avg_features.append(feature)

            if below_avg_features:
                st.write("Summary of Areas for Improvement:")
                st.write(f"Your performance is below average in the following areas: {', '.join(below_avg_features)}.")
            else:
                st.write("All your metrics are above or at the average level.")

            # Bar Chart Comparison using Plotly
            fig = go.Figure(data=[
                go.Bar(name='Student Value', x=features, y=student_values),
                go.Bar(name='Average Value', x=features, y=average_values)
            ])
            fig.update_layout(barmode='group', title="Feature Comparison", yaxis_title="Value")
            st.plotly_chart(fig)

            if below_avg_features: #Only show detailed breakdown if there are below average features
                st.write("Detailed Breakdown:")
                for i, feature in enumerate(features):
                    st.write(f"- Your **{feature}** ({student_values[i]:.2f}) is {comparison_texts[i]} compared to the average ({average_values[i]:.2f}).")
                    if comparison_texts[i] == "below average":
                        st.write(f"  * Consider focusing on improving {feature} to enhance your project completion chances.")

            st.write("Additional Suggestions:")
            st.write("- Seek extra help from instructors or tutors.")
            st.write("- Review course materials regularly.")
            st.write("- Form study groups with classmates.")
            st.write("- Improve time management skills.")

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
            df = pd.read_csv(uploaded_file)
            show_process = st.checkbox("Show Data Processing Steps", value=True)
            models, x_test, y_test, input_cols, original_df = train_models(df, show_process)
            st.write("Models trained successfully!")

            feature_means = original_df[input_cols].mean()

            new_data_file = st.file_uploader("Upload CSV (New Student Data)", type=["csv"])
            if new_data_file is not None:
                new_df = pd.read_csv(new_data_file)
                if 'Project_complete' in new_df.columns:
                    new_df = new_df.drop(columns=['Project_complete'])

                student_data_list = new_df.to_dict(orient='records')
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

                    analyze_results(np.array(all_predictions), student_data_list, feature_means)

                    # plot_roc_curve(model, x_test, y_test, selected_model)
                    # st.pyplot()

                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                st.info("Please upload new student data for predictions.")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")

if __name__ == "__main__":
    main()