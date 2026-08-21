import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)
from config.settings import settings
from core.data_pipeline import DataPipeline

class MLEngine:
    @staticmethod
    def train_classification_models():
        X_train, X_test, y_train, y_test = DataPipeline.get_classification_split()

        # Ensure target is integer binary (0 and 1)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), settings.NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), settings.CATEGORICAL_COLS)
        ])

        pipelines = {
            "Random Forest (Balanced)": Pipeline([
                ('pre', preprocessor),
                ('clf', RandomForestClassifier(n_estimators=75, max_depth=10, class_weight='balanced', random_state=settings.RANDOM_STATE, n_jobs=-1))
            ]),
            "Logistic Regression": Pipeline([
                ('pre', preprocessor),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=settings.RANDOM_STATE))
            ]),
            "Gradient Boosting": Pipeline([
                ('pre', preprocessor),
                ('clf', GradientBoostingClassifier(n_estimators=60, max_depth=4, random_state=settings.RANDOM_STATE))
            ])
        }

        metrics = []
        trained = {}
        predictions = {}
        probabilities = {}

        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            
            # Safe probability indexing
            probs = pipe.predict_proba(X_test)
            if probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                y_prob = probs[:, 0]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Safe ROC-AUC calculation
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = 0.5

            metrics.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(auc, 4)
            })
            trained[name] = pipe
            predictions[name] = y_pred
            probabilities[name] = y_prob

        df_metrics = pd.DataFrame(metrics).sort_values(by="F1-Score", ascending=False)
        return df_metrics, trained, y_test, predictions, probabilities

    @staticmethod
    def train_regression_models():
        Xr_train, Xr_test, yr_train, yr_test = DataPipeline.get_regression_split()

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), settings.REGRESSION_NUM_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), settings.CATEGORICAL_COLS)
        ])

        pipelines = {
            "Ridge Regression": Pipeline([('pre', preprocessor), ('reg', Ridge(alpha=1.0))]),
            "Linear Regression": Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
        }

        metrics = []
        trained = {}
        predictions = {}

        for name, pipe in pipelines.items():
            pipe.fit(Xr_train, yr_train)
            yp = pipe.predict(Xr_test)

            r2 = r2_score(yr_test, yp)
            rmse = np.sqrt(mean_squared_error(yr_test, yp))
            mae = mean_absolute_error(yr_test, yp)

            metrics.append({
                "Algorithm": name,
                "R-Squared (R²)": round(r2, 4),
                "RMSE (mg/dL)": round(rmse, 2),
                "MAE (mg/dL)": round(mae, 2)
            })
            trained[name] = pipe
            predictions[name] = yp

        df_metrics = pd.DataFrame(metrics).sort_values(by="R-Squared (R²)", ascending=False)
        return df_metrics, trained, yr_test, predictions