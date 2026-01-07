"""
Sistema de Analisis de Riesgo Crediticio
Aplicacion Flask para analisis de riesgo de prestamos existentes
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import io

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credit_risk_evaluator import (
    analyze_loan_risk,
    load_model,
    validate_input,
    load_test_dataset,
    VALID_VALUES
)

app = Flask(__name__)

# Cargar modelo al iniciar (pre-carga en memoria)
print("Cargando modelo de riesgo crediticio...")
load_model()  # Pre-carga el modelo en memoria
print("Modelo cargado exitosamente!")

# Cache para datos de test evaluados
TEST_DATA_CACHE = {
    'data': None,
    'predictions': None,
    'timestamp': None,
    'summary': None
}


# ==================== RUTAS DE PAGINAS ====================

@app.route('/')
def index():
    """Pagina principal - Dashboard"""
    return render_template('dashboard.html')


@app.route('/dashboard')
def dashboard():
    """Alias para Dashboard"""
    return render_template('dashboard.html')


@app.route('/evaluacion')
def evaluacion():
    """Pagina de evaluacion individual de prestamos"""
    return render_template('evaluacion_new.html', feature_config=VALID_VALUES)


@app.route('/test-data')
def test_data_view():
    """Redirige a cartera"""
    return render_template('cartera.html')


@app.route('/cartera')
def cartera():
    """Pagina de cartera de prestamos (evaluacion masiva)"""
    return render_template('cartera.html')


@app.route('/lote')
def lote():
    """Pagina de evaluacion por lotes"""
    return render_template('lote.html')


@app.route('/historial')
def historial():
    """Pagina de historial de evaluaciones"""
    return render_template('historial.html')


@app.route('/configuracion')
def configuracion():
    """Pagina de configuracion del sistema"""
    return render_template('configuracion.html')


# ==================== API ENDPOINTS ====================

@app.route('/api/evaluar', methods=['POST'])
def api_evaluar():
    """
    API para evaluar el riesgo de un prestamo individual
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos'
            }), 400
        
        # Validar entrada
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Analizar riesgo
        result = analyze_loan_risk(data)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error al procesar la solicitud: {str(e)}'
        }), 500


@app.route('/api/load-test-data', methods=['GET'])
def api_load_test_data():
    """
    Carga y evalua los datos de prueba del dataset
    """
    try:
        global TEST_DATA_CACHE
        
        # Verificar si hay datos en cache (menos de 1 hora)
        if TEST_DATA_CACHE['data'] is not None and TEST_DATA_CACHE['timestamp'] is not None:
            cache_age = (datetime.now() - TEST_DATA_CACHE['timestamp']).seconds
            if cache_age < 3600:
                return jsonify({
                    'success': True,
                    'data': TEST_DATA_CACHE['data'],
                    'predictions': TEST_DATA_CACHE['predictions'],
                    'summary': TEST_DATA_CACHE['summary'],
                    'from_cache': True
                })
        
        # Cargar datos de test (sin etiquetas visible al usuario)
        test_df, test_labels = load_test_dataset()
        
        if test_df is None or len(test_df) == 0:
            return jsonify({
                'success': False,
                'error': 'No se pudieron cargar los datos de prueba'
            }), 500
        
        # Tomar muestra de 500 registros
        sample_size = min(500, len(test_df))
        test_df_sample = test_df.sample(n=sample_size, random_state=42)
        
        # Evaluar cada registro
        predictions = []
        for idx, row in test_df_sample.iterrows():
            loan_data = row.to_dict()
            result = analyze_loan_risk(loan_data)
            predictions.append({
                'index': int(idx),
                'prediction': result['prediction'],
                'risk_level': result['risk_level'],
                'probability_default': result['probability_default'],
                'probability_payment': result['probability_payment'],
                'confidence': result['confidence'],
                'loan_info': result.get('loan_summary', {})
            })
        
        # Calcular resumen
        df_predictions = pd.DataFrame(predictions)
        summary = {
            'total_records': len(predictions),
            'predicted_defaults': int((df_predictions['prediction'] == 1).sum()),
            'predicted_payments': int((df_predictions['prediction'] == 0).sum()),
            'default_rate': float((df_predictions['prediction'] == 1).mean() * 100),
            'avg_default_probability': float(df_predictions['probability_default'].mean() * 100),
            'risk_distribution': df_predictions['risk_level'].value_counts().to_dict(),
            'high_risk_count': int((df_predictions['risk_level'].isin(['ALTO'])).sum())
        }
        
        # Guardar en cache
        TEST_DATA_CACHE['data'] = test_df_sample.to_dict(orient='records')
        TEST_DATA_CACHE['predictions'] = predictions
        TEST_DATA_CACHE['summary'] = summary
        TEST_DATA_CACHE['timestamp'] = datetime.now()
        
        return jsonify({
            'success': True,
            'data': TEST_DATA_CACHE['data'],
            'predictions': predictions,
            'summary': summary,
            'from_cache': False
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error al cargar datos de prueba: {str(e)}'
        }), 500


def clean_nan_values(obj):
    """Limpia valores NaN/Inf de un objeto para JSON serialization"""
    import math
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif obj is None:
        return 0
    return obj


@app.route('/api/evaluar-lote', methods=['POST'])
def api_evaluar_lote():
    """
    API para evaluar multiples prestamos desde un archivo CSV
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se recibio ningun archivo'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nombre de archivo vacio'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'El archivo debe ser CSV'
            }), 400
        
        # Leer CSV
        df = pd.read_csv(file)
        
        # Reemplazar NaN con valores por defecto
        df = df.fillna({
            'person_age': 30,
            'person_income': 50000,
            'person_emp_length': 5,
            'loan_amnt': 10000,
            'loan_int_rate': 10.0,
            'loan_percent_income': 0.2,
            'cb_person_cred_hist_length': 5,
            'person_home_ownership': 'RENT',
            'loan_intent': 'PERSONAL',
            'loan_grade': 'C',
            'cb_person_default_on_file': 'N'
        })
        
        # Validar columnas requeridas
        required_cols = ['person_age', 'person_income', 'person_home_ownership', 
                        'person_emp_length', 'loan_intent', 'loan_grade', 
                        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                        'cb_person_default_on_file', 'cb_person_cred_hist_length']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return jsonify({
                'success': False,
                'error': f'Columnas faltantes: {", ".join(missing_cols)}'
            }), 400
        
        # Evaluar cada registro
        results = []
        errors = []
        
        for idx, row in df.iterrows():
            try:
                loan_data = row.to_dict()
                # Extraer nombre del cliente si existe
                client_name = loan_data.pop('client_name', None) if 'client_name' in loan_data else None
                
                # Limpiar valores NaN en loan_data
                for key, value in loan_data.items():
                    if pd.isna(value):
                        if key in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 
                                   'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']:
                            loan_data[key] = 0
                        else:
                            loan_data[key] = 'N' if key == 'cb_person_default_on_file' else 'RENT'
                
                result = analyze_loan_risk(loan_data)
                
                # Calcular credit score basado en probabilidad
                prob_default = result.get('probability_default', 0.5)
                credit_score = int(850 - (prob_default * 550))  # Score entre 300-850
                
                # Determinar categoria del score
                if credit_score >= 740:
                    score_category = 'Excelente'
                elif credit_score >= 670:
                    score_category = 'Bueno'
                elif credit_score >= 580:
                    score_category = 'Regular'
                else:
                    score_category = 'Malo'
                
                # Calcular monto aprobado y tasa sugerida
                risk_level = result.get('risk_level', 'ALTO')
                loan_amount = float(loan_data.get('loan_amnt', 0))
                base_rate = float(loan_data.get('loan_int_rate', 10))
                
                if risk_level == 'BAJO':
                    approved_amount = loan_amount
                    suggested_rate = max(base_rate - 1, 5)
                    decision = 'APROBADO'
                elif risk_level == 'MEDIO':
                    approved_amount = loan_amount * 0.8
                    suggested_rate = base_rate + 2
                    decision = 'APROBADO CON CONDICIONES'
                else:
                    approved_amount = 0
                    suggested_rate = 0
                    decision = 'RECHAZADO'
                
                results.append({
                    'row': idx + 1,
                    'name': client_name or f'Cliente #{idx + 1}',
                    'age': int(loan_data.get('person_age', 0)),
                    'income': float(loan_data.get('person_income', 0)),
                    'loan_amount': loan_amount,
                    'credit_score': credit_score,
                    'score_category': score_category,
                    'approved_amount': round(approved_amount, 2),
                    'suggested_rate': round(suggested_rate, 2),
                    'decision': decision,
                    **result
                })
            except Exception as row_error:
                errors.append({
                    'row': idx + 1,
                    'errors': [str(row_error)]
                })
        
        if not results:
            return jsonify({
                'success': False,
                'error': 'No se pudo procesar ningún registro',
                'errors': errors
            }), 400
        
        # Calcular estadisticas
        df_results = pd.DataFrame(results)
        
        # Metricas para el frontend
        approved = len([r for r in results if r.get('decision') == 'APROBADO'])
        conditional = len([r for r in results if r.get('decision') == 'APROBADO CON CONDICIONES'])
        rejected = len([r for r in results if r.get('decision') == 'RECHAZADO'])
        total_requested = sum(r.get('loan_amount', 0) for r in results)
        total_approved = sum(r.get('approved_amount', 0) for r in results)
        
        metrics = {
            'total_processed': len(results),
            'approved': approved,
            'conditional': conditional,
            'rejected': rejected,
            'approval_rate': round((approved + conditional) / len(results) * 100, 1) if results else 0,
            'total_requested': total_requested,
            'total_approved': total_approved
        }
        
        stats = {
            'total': len(results),
            'predicted_defaults': int((df_results['prediction'] == 1).sum()),
            'predicted_payments': int((df_results['prediction'] == 0).sum()),
            'default_rate': float((df_results['prediction'] == 1).mean() * 100),
            'avg_default_probability': float(df_results['probability_default'].mean() * 100),
            'risk_distribution': df_results['risk_level'].value_counts().to_dict()
        }
        
        # Limpiar NaN antes de serializar
        response_data = clean_nan_values({
            'success': True,
            'results': results,
            'metrics': metrics,
            'stats': stats,
            'errors': errors if errors else [],
            'batch_id': datetime.now().strftime('%Y%m%d%H%M%S')
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error al procesar archivo: {str(e)}'
        }), 500


@app.route('/api/dashboard-metrics', methods=['GET'])
def api_dashboard_metrics():
    """
    Retorna metricas para el dashboard principal
    """
    try:
        # Obtener config del modelo
        _, feature_config = load_model()
        num_features = len(feature_config.get('numerical_features', [])) + len(feature_config.get('categorical_features', []))
        
        metrics = {
            'model_info': {
                'type': 'LightGBM Classifier',
                'features': num_features,
                'status': 'Activo'
            },
            'test_data_loaded': TEST_DATA_CACHE['data'] is not None,
            'test_data_summary': TEST_DATA_CACHE.get('summary', None)
        }
        
        # Si hay datos de test, agregar metricas adicionales
        if TEST_DATA_CACHE['predictions'] is not None:
            predictions = TEST_DATA_CACHE['predictions']
            df_pred = pd.DataFrame(predictions)
            
            # Distribucion por nivel de riesgo
            risk_dist = df_pred['risk_level'].value_counts().to_dict()
            
            # Distribucion de probabilidades
            prob_ranges = {
                '0-20%': int(((df_pred['probability_default'] >= 0) & (df_pred['probability_default'] < 0.2)).sum()),
                '20-40%': int(((df_pred['probability_default'] >= 0.2) & (df_pred['probability_default'] < 0.4)).sum()),
                '40-60%': int(((df_pred['probability_default'] >= 0.4) & (df_pred['probability_default'] < 0.6)).sum()),
                '60-80%': int(((df_pred['probability_default'] >= 0.6) & (df_pred['probability_default'] < 0.8)).sum()),
                '80-100%': int((df_pred['probability_default'] >= 0.8).sum())
            }
            
            metrics['risk_distribution'] = risk_dist
            metrics['probability_distribution'] = prob_ranges
            metrics['total_evaluated'] = len(predictions)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-predictions', methods=['GET'])
def api_export_predictions():
    """
    Exporta las predicciones de los datos de prueba como CSV
    """
    try:
        if TEST_DATA_CACHE['predictions'] is None:
            return jsonify({
                'success': False,
                'error': 'No hay predicciones disponibles. Cargue los datos de prueba primero.'
            }), 400
        
        # Crear DataFrame con predicciones
        df_data = pd.DataFrame(TEST_DATA_CACHE['data'])
        df_pred = pd.DataFrame(TEST_DATA_CACHE['predictions'])
        
        # Combinar datos originales con predicciones
        export_df = df_data.copy()
        export_df['predicted_default'] = df_pred['prediction']
        export_df['risk_level'] = df_pred['risk_level']
        export_df['probability_default'] = df_pred['probability_default'].round(4)
        export_df['probability_payment'] = df_pred['probability_payment'].round(4)
        export_df['confidence'] = df_pred['confidence'].round(4)
        
        # Convertir a CSV
        output = io.StringIO()
        export_df.to_csv(output, index=False)
        output.seek(0)
        
        # Crear respuesta
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'predicciones_riesgo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/feature-config', methods=['GET'])
def api_feature_config():
    """Retorna la configuracion de features para el formulario"""
    _, feature_config = load_model()
    return jsonify({
        'success': True,
        'config': feature_config,
        'valid_values': VALID_VALUES
    })


@app.route('/api/health', methods=['GET'])
def api_health():
    """Endpoint de salud del servicio"""
    pipeline, config = load_model()
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline is not None,
        'config_loaded': config is not None,
        'timestamp': datetime.now().isoformat()
    })


# ==================== MAIN ====================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
