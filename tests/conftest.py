"""
Shared pytest fixtures for GBRS test suite.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


@pytest.fixture(scope="session")
def diabetes_data():
    """Load sklearn diabetes dataset for regression tests."""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": data.feature_names,
    }


@pytest.fixture(scope="session")
def breast_cancer_data():
    """Load sklearn breast cancer dataset for classification tests."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": data.feature_names,
    }


@pytest.fixture(scope="session")
def veteran_data():
    """Load veteran lung cancer dataset for survival tests."""
    try:
        from lifelines.datasets import load_veterans_lung_cancer

        data = load_veterans_lung_cancer()

        # Extract features (excluding time and status)
        feature_cols = ["trt", "celltype", "karno", "diagtime", "age", "prior"]
        X = data[feature_cols].values

        # One-hot encode celltype if needed
        if data["celltype"].dtype == "object":
            celltype_dummies = pd.get_dummies(data["celltype"], prefix="celltype")
            X = np.column_stack(
                [
                    data[["trt", "karno", "diagtime", "age", "prior"]].values,
                    celltype_dummies.values,
                ]
            )

        time = data["time"].values
        event = data["status"].values

        # Train/test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, time_train, time_test, event_train, event_test = (
            train_test_split(X, time, event, test_size=0.2, random_state=42)
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "time_train": time_train,
            "time_test": time_test,
            "event_train": event_train,
            "event_test": event_test,
        }
    except ImportError:
        pytest.skip("lifelines not installed")


@pytest.fixture(scope="session")
def linear_baseline(diabetes_data):
    """Fitted LinearRegression baseline for comparison."""
    model = LinearRegression()
    model.fit(diabetes_data["X_train"], diabetes_data["y_train"])
    return model


@pytest.fixture(scope="session")
def logistic_baseline(breast_cancer_data):
    """Fitted LogisticRegression baseline for comparison."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(breast_cancer_data["X_train"], breast_cancer_data["y_train"])
    return model


@pytest.fixture(scope="session")
def cox_baseline(veteran_data):
    """Fitted CoxPH baseline for comparison."""
    try:
        from lifelines import CoxPHFitter
        import pandas as pd

        # Create DataFrame for lifelines
        df_train = pd.DataFrame(veteran_data["X_train"])
        df_train["time"] = veteran_data["time_train"]
        df_train["event"] = veteran_data["event_train"]

        model = CoxPHFitter()
        model.fit(df_train, duration_col="time", event_col="event")
        return model
    except ImportError:
        pytest.skip("lifelines not installed")


def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def calculate_c_index(time, event, risk_scores):
    """
    Calculate Harrell's C-index for survival analysis.

    Args:
        time: Survival times
        event: Event indicators (1=event, 0=censored)
        risk_scores: Predicted risk scores (higher = higher risk)

    Returns:
        C-index value between 0 and 1
    """
    n = len(time)
    concordant = 0
    permissible = 0

    for i in range(n):
        if event[i] == 1:  # Only consider uncensored cases
            for j in range(n):
                if time[i] < time[j]:  # i died before j
                    permissible += 1
                    # Concordant if i has higher risk than j
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        concordant += 0.5

    if permissible == 0:
        return 0.5

    return concordant / permissible
