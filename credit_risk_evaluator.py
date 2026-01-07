"""
Sistema de Analisis de Riesgo Crediticio - Produccion
=====================================================

Este modulo analiza el riesgo de impago (default) de prestamos existentes.
El modelo predice si un prestamo que YA fue otorgado sera pagado o no.

IMPORTANTE: Este sistema NO decide si aprobar prestamos, sino que evalua
el riesgo de impago de prestamos ya realizados.

Variable objetivo: loan_status (0 = Pagara, 1 = No pagara/Default)

Uso:
    from credit_risk_evaluator import analyze_loan_risk, validate_input
    
    result = analyze_loan_risk(loan_data)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any

MODEL_DIR = Path(__file__).parent / "models"
DATASET_DIR = Path(__file__).parent / "dataset"

_pipeline = None
_feature_config = None
_test_data = None
_test_labels = None


def load_model():
    """Carga el modelo y configuracion en memoria"""
    global _pipeline, _feature_config
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_DIR / "credit_risk_pipeline.joblib")
        _feature_config = joblib.load(MODEL_DIR / "feature_config.joblib")
    return _pipeline, _feature_config


def load_test_dataset():
    """Carga el dataset de test (prestamos reales para evaluar)."""
    global _test_data, _test_labels
    if _test_data is not None:
        return _test_data, _test_labels
    
    try:
        df = pd.read_csv(DATASET_DIR / "credit_risk_dataset.csv")
        
        df_clean = df.dropna().copy()
        df_clean = df_clean[df_clean['person_age'] <= 80]
        df_clean = df_clean[df_clean['person_emp_length'] <= 60]
        
        df_clean['loan_to_income_ratio'] = df_clean['loan_amnt'] / df_clean['person_income']
        df_clean['income_per_year_employed'] = df_clean['person_income'] / (df_clean['person_emp_length'] + 1)
        
        from sklearn.model_selection import train_test_split
        _, test_data, _, test_labels = train_test_split(
            df_clean.drop('loan_status', axis=1),
            df_clean['loan_status'],
            test_size=0.2,
            random_state=42,
            stratify=df_clean['loan_status']
        )
        
        _test_data = test_data.reset_index(drop=True)
        _test_labels = test_labels.reset_index(drop=True)
        return _test_data, _test_labels
    
    except Exception as e:
        print(f"Error cargando dataset: {e}")
        return None, None


def analyze_loan_risk(loan_data: dict) -> dict:
    """Analiza el riesgo de impago de un prestamo."""
    pipeline, feature_config = load_model()
    
    input_copy = loan_data.copy()
    
    if 'loan_to_income_ratio' not in input_copy:
        input_copy['loan_to_income_ratio'] = input_copy['loan_amnt'] / input_copy['person_income']
    if 'income_per_year_employed' not in input_copy:
        input_copy['income_per_year_employed'] = input_copy['person_income'] / (input_copy['person_emp_length'] + 1)
    
    df_input = pd.DataFrame([input_copy])
    df_input = df_input[feature_config['categorical_features'] + feature_config['numerical_features']]
    
    prediction = pipeline.predict(df_input)[0]
    probability = pipeline.predict_proba(df_input)[0]
    prob_payment = probability[0]
    prob_default = probability[1]
    
    if prob_default < 0.3:
        risk_level = "BAJO"
        risk_color = "#22c55e"
    elif prob_default < 0.6:
        risk_level = "MEDIO"
        risk_color = "#f59e0b"
    else:
        risk_level = "ALTO"
        risk_color = "#ef4444"
    
    risk_factors = _identify_risk_factors(loan_data, prob_default)
    
    loan_summary = {
        "monto": loan_data['loan_amnt'],
        "tasa_interes": loan_data['loan_int_rate'],
        "proposito": _translate_loan_intent(loan_data['loan_intent']),
        "grado": loan_data['loan_grade'],
        "ratio_ingreso": round(loan_data['loan_amnt'] / loan_data['person_income'], 3)
    }
    
    debtor_info = {
        "edad": loan_data['person_age'],
        "ingreso_anual": loan_data['person_income'],
        "antiguedad_laboral": loan_data['person_emp_length'],
        "tipo_vivienda": _translate_home_ownership(loan_data['person_home_ownership']),
        "historial_default": "Si" if loan_data['cb_person_default_on_file'] == 'Y' else "No",
        "historial_crediticio_anos": loan_data['cb_person_cred_hist_length']
    }
    
    return {
        "prediction": int(prediction),
        "prediction_label": "DEFAULT (No pagara)" if prediction == 1 else "PAGARA",
        "risk_level": risk_level,
        "risk_color": risk_color,
        "probability_default": round(prob_default, 4),
        "probability_payment": round(prob_payment, 4),
        "confidence": round(max(probability), 4),
        "risk_factors": risk_factors,
        "loan_summary": loan_summary,
        "debtor_info": debtor_info
    }


def _identify_risk_factors(loan_data: dict, prob_default: float) -> List[dict]:
    """Identifica los principales factores que contribuyen al riesgo"""
    factors = []
    
    if loan_data.get('cb_person_default_on_file') == 'Y':
        factors.append({
            "factor": "Historial de incumplimiento",
            "descripcion": "El deudor tiene registro previo de impago",
            "impacto": "ALTO",
            "color": "#ef4444"
        })
    
    grade = loan_data.get('loan_grade', 'A')
    if grade in ['E', 'F', 'G']:
        factors.append({
            "factor": f"Grado de prestamo {grade}",
            "descripcion": "Prestamo clasificado como alto riesgo por el originador",
            "impacto": "ALTO",
            "color": "#ef4444"
        })
    elif grade in ['C', 'D']:
        factors.append({
            "factor": f"Grado de prestamo {grade}",
            "descripcion": "Prestamo con riesgo moderado",
            "impacto": "MEDIO",
            "color": "#f59e0b"
        })
    
    loan_percent = loan_data.get('loan_percent_income', 0)
    if loan_percent > 0.4:
        factors.append({
            "factor": "Alto ratio prestamo/ingreso",
            "descripcion": f"El prestamo representa {loan_percent*100:.1f}% del ingreso anual",
            "impacto": "ALTO",
            "color": "#ef4444"
        })
    elif loan_percent > 0.25:
        factors.append({
            "factor": "Ratio prestamo/ingreso moderado",
            "descripcion": f"El prestamo representa {loan_percent*100:.1f}% del ingreso anual",
            "impacto": "MEDIO",
            "color": "#f59e0b"
        })
    
    int_rate = loan_data.get('loan_int_rate', 0)
    if int_rate > 18:
        factors.append({
            "factor": "Tasa de interes elevada",
            "descripcion": f"Tasa del {int_rate}% indica riesgo percibido alto",
            "impacto": "ALTO",
            "color": "#ef4444"
        })
    elif int_rate > 13:
        factors.append({
            "factor": "Tasa de interes moderada-alta",
            "descripcion": f"Tasa del {int_rate}%",
            "impacto": "MEDIO",
            "color": "#f59e0b"
        })
    
    emp_length = loan_data.get('person_emp_length', 0)
    if emp_length < 1:
        factors.append({
            "factor": "Poca estabilidad laboral",
            "descripcion": "Menos de 1 anio en empleo actual",
            "impacto": "ALTO",
            "color": "#ef4444"
        })
    elif emp_length < 2:
        factors.append({
            "factor": "Antiguedad laboral limitada",
            "descripcion": f"{emp_length} anios en empleo actual",
            "impacto": "MEDIO",
            "color": "#f59e0b"
        })
    
    cred_hist = loan_data.get('cb_person_cred_hist_length', 0)
    if cred_hist < 2:
        factors.append({
            "factor": "Historial crediticio corto",
            "descripcion": f"Solo {cred_hist} anios de historial",
            "impacto": "MEDIO",
            "color": "#f59e0b"
        })
    
    age = loan_data.get('person_age', 30)
    if age < 25:
        factors.append({
            "factor": "Edad joven",
            "descripcion": "Perfiles jovenes tienen estadisticamente mayor riesgo",
            "impacto": "BAJO",
            "color": "#3b82f6"
        })
    
    if prob_default < 0.3:
        if loan_data.get('cb_person_default_on_file') == 'N':
            factors.append({
                "factor": "Sin historial de incumplimiento",
                "descripcion": "El deudor nunca ha tenido defaults registrados",
                "impacto": "POSITIVO",
                "color": "#22c55e"
            })
        
        if grade in ['A', 'B']:
            factors.append({
                "factor": f"Excelente grado de prestamo ({grade})",
                "descripcion": "Prestamo clasificado como bajo riesgo",
                "impacto": "POSITIVO",
                "color": "#22c55e"
            })
        
        if emp_length >= 5:
            factors.append({
                "factor": "Alta estabilidad laboral",
                "descripcion": f"{emp_length} anios en empleo actual",
                "impacto": "POSITIVO",
                "color": "#22c55e"
            })
    
    return factors


def _translate_loan_intent(intent: str) -> str:
    translations = {
        'PERSONAL': 'Personal',
        'EDUCATION': 'Educacion',
        'MEDICAL': 'Medico',
        'VENTURE': 'Negocio',
        'HOMEIMPROVEMENT': 'Mejoras del hogar',
        'DEBTCONSOLIDATION': 'Consolidacion de deuda'
    }
    return translations.get(intent, intent)


def _translate_home_ownership(ownership: str) -> str:
    translations = {
        'RENT': 'Alquiler',
        'OWN': 'Propia',
        'MORTGAGE': 'Hipoteca',
        'OTHER': 'Otro'
    }
    return translations.get(ownership, ownership)


def analyze_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Analiza un lote de prestamos y retorna predicciones con metricas agregadas."""
    pipeline, feature_config = load_model()
    
    df_copy = df.copy()
    
    if 'loan_to_income_ratio' not in df_copy.columns:
        df_copy['loan_to_income_ratio'] = df_copy['loan_amnt'] / df_copy['person_income']
    if 'income_per_year_employed' not in df_copy.columns:
        df_copy['income_per_year_employed'] = df_copy['person_income'] / (df_copy['person_emp_length'] + 1)
    
    X = df_copy[feature_config['categorical_features'] + feature_config['numerical_features']]
    
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    
    df_copy['prediction'] = predictions
    df_copy['probability_default'] = probabilities[:, 1]
    df_copy['probability_payment'] = probabilities[:, 0]
    df_copy['risk_level'] = pd.cut(
        df_copy['probability_default'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['BAJO', 'MEDIO', 'ALTO']
    )
    
    total = len(df_copy)
    predicted_defaults = (df_copy['prediction'] == 1).sum()
    predicted_payments = (df_copy['prediction'] == 0).sum()
    
    risk_distribution = df_copy['risk_level'].value_counts().to_dict()
    
    metrics = {
        "total_loans": total,
        "predicted_defaults": int(predicted_defaults),
        "predicted_payments": int(predicted_payments),
        "default_rate": round(predicted_defaults / total * 100, 2) if total > 0 else 0,
        "avg_probability_default": round(df_copy['probability_default'].mean() * 100, 2),
        "risk_distribution": {
            "bajo": int(risk_distribution.get('BAJO', 0)),
            "medio": int(risk_distribution.get('MEDIO', 0)),
            "alto": int(risk_distribution.get('ALTO', 0))
        },
        "total_amount_at_risk": round(df_copy[df_copy['prediction'] == 1]['loan_amnt'].sum(), 2),
        "total_amount_analyzed": round(df_copy['loan_amnt'].sum(), 2),
        "avg_loan_amount": round(df_copy['loan_amnt'].mean(), 2),
        "grade_distribution": df_copy['loan_grade'].value_counts().to_dict()
    }
    
    return df_copy, metrics


def get_portfolio_analysis():
    """Analiza el portafolio completo de prestamos de test."""
    test_data, test_labels = load_test_dataset()
    if test_data is None:
        return None
    
    results_df, metrics = analyze_batch(test_data)
    
    metrics['by_grade'] = {}
    for grade in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        grade_data = results_df[results_df['loan_grade'] == grade]
        if len(grade_data) > 0:
            metrics['by_grade'][grade] = {
                'count': len(grade_data),
                'default_rate': round((grade_data['prediction'] == 1).mean() * 100, 2),
                'avg_prob_default': round(grade_data['probability_default'].mean() * 100, 2),
                'total_amount': round(grade_data['loan_amnt'].sum(), 2)
            }
    
    metrics['by_intent'] = {}
    for intent in results_df['loan_intent'].unique():
        intent_data = results_df[results_df['loan_intent'] == intent]
        metrics['by_intent'][_translate_loan_intent(intent)] = {
            'count': len(intent_data),
            'default_rate': round((intent_data['prediction'] == 1).mean() * 100, 2),
            'avg_prob_default': round(intent_data['probability_default'].mean() * 100, 2)
        }
    
    return {
        'metrics': metrics,
        'sample_data': results_df.head(100).to_dict('records'),
        'full_data': results_df,
        'total_records': len(results_df)
    }


VALID_VALUES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["Y", "N"]
}


def validate_input(input_data: dict) -> Tuple[bool, List[str]]:
    """Valida los datos de entrada de un prestamo"""
    errors = []
    
    required_fields = [
        "person_age", "person_income", "person_home_ownership", "person_emp_length",
        "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
        "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length"
    ]
    
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Campo requerido faltante: {field}")
    
    if errors:
        return False, errors
    
    if not (18 <= input_data["person_age"] <= 100):
        errors.append("person_age debe estar entre 18 y 100")
    
    if input_data["person_income"] <= 0:
        errors.append("person_income debe ser mayor a 0")
    
    if input_data["loan_amnt"] <= 0:
        errors.append("loan_amnt debe ser mayor a 0")
    
    for field, valid_values in VALID_VALUES.items():
        if input_data.get(field) not in valid_values:
            errors.append(f"{field} debe ser uno de: {valid_values}")
    
    return len(errors) == 0, errors


def predict_credit_risk(input_data: dict) -> dict:
    """Funcion de compatibilidad."""
    result = analyze_loan_risk(input_data)
    return {
        "prediction": result["prediction"],
        "risk_level": result["risk_level"],
        "probability_default": result["probability_default"],
        "probability_no_default": result["probability_payment"],
        "confidence": result["confidence"]
    }


def evaluate_credit_application(input_data: dict, **kwargs) -> dict:
    """Alias para compatibilidad"""
    return analyze_loan_risk(input_data)


if __name__ == "__main__":
    test_loan = {
        "person_age": 35,
        "person_income": 75000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 8,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "B",
        "loan_amnt": 20000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.27,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7
    }
    
    is_valid, errors = validate_input(test_loan)
    if is_valid:
        result = analyze_loan_risk(test_loan)
        print(f"Prediccion: {result['prediction_label']}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Probabilidad de Default: {result['probability_default']*100:.1f}%")
    else:
        print("Errores:", errors)
