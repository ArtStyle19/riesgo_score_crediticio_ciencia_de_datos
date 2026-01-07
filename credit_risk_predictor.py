
"""Módulo de predicción de riesgo crediticio para producción"""

import pandas as pd
import joblib
from pathlib import Path

# Ruta al directorio de modelos
MODEL_DIR = Path(__file__).parent / "models"

# Cargar artefactos una sola vez
_pipeline = None
_feature_config = None

def load_model():
    """Carga el modelo y configuración en memoria"""
    global _pipeline, _feature_config
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_DIR / "credit_risk_pipeline.joblib")
        _feature_config = joblib.load(MODEL_DIR / "feature_config.joblib")
    return _pipeline, _feature_config


def predict_credit_risk(input_data: dict) -> dict:
    """
    Predice el riesgo crediticio de un solicitante.

    Parámetros:
    -----------
    input_data : dict
        Diccionario con las características del solicitante:
        - person_age: int - Edad del solicitante
        - person_income: float - Ingreso anual
        - person_home_ownership: str - Tipo de vivienda (RENT, OWN, MORTGAGE, OTHER)
        - person_emp_length: float - Años de empleo
        - loan_intent: str - Propósito (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
        - loan_grade: str - Grado del préstamo (A-G)
        - loan_amnt: float - Monto del préstamo
        - loan_int_rate: float - Tasa de interés
        - loan_percent_income: float - Porcentaje préstamo/ingreso
        - cb_person_default_on_file: str - Historial de default (Y, N)
        - cb_person_cred_hist_length: int - Años de historial crediticio

    Retorna:
    --------
    dict: Resultado de la predicción con:
        - prediction: 0 o 1
        - risk_level: ALTO o BAJO
        - probability_default: float
        - probability_no_default: float
        - recommendation: APROBAR o RECHAZAR
        - confidence: float
    """
    pipeline, feature_config = load_model()

    # Calcular características derivadas
    input_data = input_data.copy()
    input_data["loan_to_income_ratio"] = input_data["loan_amnt"] / input_data["person_income"]
    input_data["income_per_year_employed"] = input_data["person_income"] / (input_data["person_emp_length"] + 1)

    # Crear DataFrame con orden correcto de columnas
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_config["categorical_features"] + feature_config["numerical_features"]]

    # Realizar predicción
    prediction = pipeline.predict(df_input)[0]
    probability = pipeline.predict_proba(df_input)[0]

    return {
        "prediction": int(prediction),
        "risk_level": "ALTO" if prediction == 1 else "BAJO",
        "probability_no_default": round(float(probability[0]), 4),
        "probability_default": round(float(probability[1]), 4),
        "recommendation": "RECHAZAR" if probability[1] > 0.5 else "APROBAR",
        "confidence": round(float(max(probability)), 4)
    }


# Valores válidos para campos categóricos
VALID_VALUES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["Y", "N"]
}


def validate_input(input_data: dict) -> tuple:
    """
    Valida los datos de entrada.

    Retorna:
    --------
    tuple: (is_valid: bool, errors: list)
    """
    errors = []

    required_fields = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
    ]

    # Verificar campos requeridos
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Campo requerido faltante: {field}")

    if errors:
        return False, errors

    # Validar rangos numéricos
    if not (18 <= input_data["person_age"] <= 100):
        errors.append("person_age debe estar entre 18 y 100")

    if input_data["person_income"] <= 0:
        errors.append("person_income debe ser mayor a 0")

    if input_data["loan_amnt"] <= 0:
        errors.append("loan_amnt debe ser mayor a 0")

    # Validar campos categóricos
    for field, valid_values in VALID_VALUES.items():
        if input_data.get(field) not in valid_values:
            errors.append(f"{field} debe ser uno de: {valid_values}")

    return len(errors) == 0, errors


if __name__ == "__main__":
    # Ejemplo de uso
    test_data = {
        "person_age": 28,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.17,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }

    is_valid, errors = validate_input(test_data)
    if is_valid:
        result = predict_credit_risk(test_data)
        print("Resultado:", result)
    else:
        print("Errores de validación:", errors)
