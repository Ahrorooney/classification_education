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

def main():
    # Load the dataset
    df = pd.read_csv("sample.csv")

    if df is None or df.empty:
        print("Dataset is empty or not found.")
        return

    # Preprocess the dataset
    try:
        df_processed = preprocess_data(df.copy())
    except ValueError as e:
        print(f"Error during preprocessing: {e}")
        return

    # Remove outliers
    df = remove_outliers_from_data(df_processed)

    # Normalize the dataset
    df = normalize_data(df)

    # Separate features and target
    x = df.drop('Project_complete', axis=1)
    y = df['Project_complete']

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    # Train the model
    lr_model = train_and_evaluate_logistic_regression(x_train_smote, y_train_smote, x_test, y_test)
    dt_model = train_and_evaluate_decision_tree(x_train_smote, y_train_smote, x_test, y_test)
    rf_model = train_and_evaluate_random_forest(x_train_smote, y_train_smote, x_test, y_test)
    gb_model = train_and_evaluate_gradient_boosting(x_train_smote, y_train_smote, x_test, y_test)
    svm_model = train_and_evaluate_svm(x_train_smote, y_train_smote, x_test, y_test)
    knn_model = train_and_evaluate_knn(x_train_smote, y_train_smote, x_test, y_test)
    ann_model = train_and_evaluate_ann(x_train_smote, y_train_smote, x_test, y_test)

    # if lr_model is not None: plot_roc_curve(lr_model, x_test, y_test, "Logistic Regression")
    # if dt_model is not None: plot_roc_curve(dt_model, x_test, y_test, "Decision Tree")
    # if rf_model is not None: plot_roc_curve(rf_model, x_test, y_test, "Random Forest")
    # if gb_model is not None: plot_roc_curve(gb_model, x_test, y_test, "Gradient Boosting")
    # if svm_model is not None: plot_roc_curve(svm_model, x_test, y_test, "SVM")
    # if knn_model is not None: plot_roc_curve(knn_model, x_test, y_test, "KNN")
    # plot_roc_curve(ann_model, x_test, y_test, "ANN")


if __name__ == "__main__":
    main()
