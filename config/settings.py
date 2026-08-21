import os

class Settings:
    PROJECT_NAME: str = "Metabolic Intelligence ML Diagnostic Suite"
    VERSION: str = "5.0.0"
    
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH: str = os.path.join(BASE_DIR, "data", "diabetes_prediction_dataset.csv")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Feature configurations
    NUMERICAL_COLS: list = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    CATEGORICAL_COLS: list = ['gender', 'smoking_history', 'diet_quality', 'physical_activity', 'sleep_quality', 'stress_level']
    REGRESSION_NUM_COLS: list = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level']

settings = Settings()