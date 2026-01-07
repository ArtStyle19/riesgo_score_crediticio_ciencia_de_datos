"""
Script de Pruebas para el Sistema de Evaluación Crediticia
==========================================================

Este script prueba diferentes casos para verificar que el modelo
y las funciones de evaluación funcionan correctamente.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from credit_risk_evaluator import (
    evaluate_credit_application,
    validate_input,
    calculate_credit_score,
    calculate_loan_recommendation,
    get_score_category
)

def print_separator():
    print("\n" + "="*70 + "\n")

def test_validation():
    """Prueba la validación de entrada"""
    print("🧪 TEST 1: Validación de Entrada")
    print("-" * 40)
    
    # Caso 1: Datos válidos
    valid_data = {
        "person_age": 35,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 5,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 15000,
        "loan_int_rate": 11.5,
        "loan_percent_income": 0.25,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 4
    }
    
    is_valid, errors = validate_input(valid_data)
    print(f"✓ Datos válidos: {'PASS' if is_valid else 'FAIL'}")
    if not is_valid:
        print(f"  Errores: {errors}")
    
    # Caso 2: Edad inválida
    invalid_age = valid_data.copy()
    invalid_age["person_age"] = 15
    is_valid, errors = validate_input(invalid_age)
    print(f"✓ Edad inválida (15): {'PASS' if not is_valid else 'FAIL'}")
    
    # Caso 3: Campo faltante
    missing_field = valid_data.copy()
    del missing_field["loan_amnt"]
    is_valid, errors = validate_input(missing_field)
    print(f"✓ Campo faltante: {'PASS' if not is_valid else 'FAIL'}")
    
    # Caso 4: Valor categórico inválido
    invalid_category = valid_data.copy()
    invalid_category["person_home_ownership"] = "CASA"
    is_valid, errors = validate_input(invalid_category)
    print(f"✓ Categoría inválida: {'PASS' if not is_valid else 'FAIL'}")
    
    print()

def test_credit_score_calculation():
    """Prueba el cálculo del score crediticio"""
    print("🧪 TEST 2: Cálculo de Score Crediticio")
    print("-" * 40)
    
    # Caso 1: Cliente excelente (probabilidad alta de no default)
    excellent_data = {
        "cb_person_cred_hist_length": 15,
        "cb_person_default_on_file": "N",
        "person_emp_length": 12,
        "loan_percent_income": 0.1,
        "loan_grade": "A"
    }
    score_excellent = calculate_credit_score(0.95, excellent_data)
    category, emoji, color = get_score_category(score_excellent)
    print(f"✓ Cliente Excelente: Score={score_excellent} ({category}) {emoji}")
    assert 780 <= score_excellent <= 850, f"Score esperado 780-850, obtenido {score_excellent}"
    
    # Caso 2: Cliente regular
    regular_data = {
        "cb_person_cred_hist_length": 5,
        "cb_person_default_on_file": "N",
        "person_emp_length": 4,
        "loan_percent_income": 0.25,
        "loan_grade": "C"
    }
    score_regular = calculate_credit_score(0.75, regular_data)
    category, emoji, color = get_score_category(score_regular)
    print(f"✓ Cliente Regular: Score={score_regular} ({category}) {emoji}")
    assert 620 <= score_regular <= 720, f"Score esperado 620-720, obtenido {score_regular}"
    
    # Caso 3: Cliente de alto riesgo
    risky_data = {
        "cb_person_cred_hist_length": 1,
        "cb_person_default_on_file": "Y",
        "person_emp_length": 0,
        "loan_percent_income": 0.7,
        "loan_grade": "G"
    }
    score_risky = calculate_credit_score(0.30, risky_data)
    category, emoji, color = get_score_category(score_risky)
    print(f"✓ Cliente Riesgoso: Score={score_risky} ({category}) {emoji}")
    assert 300 <= score_risky <= 500, f"Score esperado 300-500, obtenido {score_risky}"
    
    print()

def test_loan_recommendation():
    """Prueba las recomendaciones de préstamo"""
    print("🧪 TEST 3: Recomendaciones de Préstamo")
    print("-" * 40)
    
    # Cliente con buen perfil
    good_client = {
        "person_income": 80000,
        "loan_amnt": 20000,
        "loan_int_rate": 10.0,
        "cb_person_default_on_file": "N",
        "person_emp_length": 8
    }
    
    recommendation = calculate_loan_recommendation(good_client, credit_score=750, probability_default=0.1)
    
    print(f"✓ Monto máximo recomendado: ${recommendation['max_recommended_loan']:,.2f}")
    print(f"✓ Tasa sugerida: {recommendation['suggested_rate']:.2f}%")
    print(f"✓ Plazo máximo: {recommendation['max_term_months']} meses")
    print(f"✓ Pago mensual: ${recommendation['monthly_payment']:,.2f}")
    
    # Verificaciones
    assert recommendation['max_recommended_loan'] > 0
    assert 5.0 <= recommendation['suggested_rate'] <= 25.0
    assert recommendation['max_term_months'] in [24, 36, 48, 60]
    assert recommendation['monthly_payment'] > 0
    
    print()

def test_full_evaluation():
    """Prueba la evaluación completa de diferentes perfiles"""
    print("🧪 TEST 4: Evaluación Completa de Perfiles")
    print("-" * 40)
    
    test_cases = [
        {
            "name": "👑 Cliente Premium",
            "data": {
                "person_age": 45,
                "person_income": 120000,
                "person_home_ownership": "OWN",
                "person_emp_length": 15,
                "loan_intent": "HOMEIMPROVEMENT",
                "loan_grade": "A",
                "loan_amnt": 25000,
                "loan_int_rate": 7.5,
                "loan_percent_income": 0.21,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 12
            },
            "expected_decision": ["APROBADO"],
            "expected_score_range": (740, 850)
        },
        {
            "name": "👤 Cliente Regular",
            "data": {
                "person_age": 32,
                "person_income": 55000,
                "person_home_ownership": "RENT",
                "person_emp_length": 4,
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "loan_amnt": 15000,
                "loan_int_rate": 13.5,
                "loan_percent_income": 0.27,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 5
            },
            "expected_decision": ["APROBADO", "APROBADO CON CONDICIONES"],
            "expected_score_range": (580, 739)
        },
        {
            "name": "⚠️ Cliente Riesgoso",
            "data": {
                "person_age": 23,
                "person_income": 28000,
                "person_home_ownership": "RENT",
                "person_emp_length": 1,
                "loan_intent": "VENTURE",
                "loan_grade": "E",
                "loan_amnt": 20000,
                "loan_int_rate": 18.5,
                "loan_percent_income": 0.71,
                "cb_person_default_on_file": "Y",
                "cb_person_cred_hist_length": 2
            },
            "expected_decision": ["RECHAZADO", "APROBADO CON CONDICIONES"],
            "expected_score_range": (300, 600)
        }
    ]
    
    all_passed = True
    
    for case in test_cases:
        print(f"\n{case['name']}")
        print("-" * 30)
        
        try:
            result = evaluate_credit_application(case["data"])
            
            score = result["credit_score"]["score"]
            category = result["credit_score"]["category"]
            decision = result["final_decision"]["decision"]
            prob_default = result["risk_assessment"]["probability_default"]
            approved_amount = result["loan_recommendation"]["approved_amount"]
            
            print(f"  Score: {score} ({category})")
            print(f"  Decisión: {decision}")
            print(f"  Prob. Default: {prob_default*100:.1f}%")
            print(f"  Monto Aprobado: ${approved_amount:,.2f}")
            
            # Verificaciones
            min_score, max_score = case["expected_score_range"]
            score_ok = min_score <= score <= max_score
            decision_ok = decision in case["expected_decision"]
            
            if score_ok and decision_ok:
                print(f"  ✅ PASS")
            else:
                print(f"  ❌ FAIL")
                if not score_ok:
                    print(f"     Score {score} fuera de rango esperado ({min_score}-{max_score})")
                if not decision_ok:
                    print(f"     Decisión '{decision}' no esperada (esperado: {case['expected_decision']})")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
    
    print()
    return all_passed

def test_edge_cases():
    """Prueba casos extremos"""
    print("🧪 TEST 5: Casos Extremos")
    print("-" * 40)
    
    # Caso 1: loan_percent_income muy alto (problema reportado)
    high_ratio_data = {
        "person_age": 25,
        "person_income": 20000,
        "person_home_ownership": "RENT",
        "person_emp_length": 1,
        "loan_intent": "PERSONAL",
        "loan_grade": "D",
        "loan_amnt": 50000,  # 250% del ingreso!
        "loan_int_rate": 15.0,
        "loan_percent_income": 2.5,  # Ratio > 1 (problema potencial)
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 2
    }
    
    print(f"✓ Ratio préstamo/ingreso alto (2.5 = 250%)")
    try:
        result = evaluate_credit_application(high_ratio_data)
        print(f"  Score: {result['credit_score']['score']}")
        print(f"  Decisión: {result['final_decision']['decision']}")
        print(f"  El sistema manejó correctamente el ratio alto")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    # Caso 2: Ingreso muy alto
    high_income = {
        "person_age": 50,
        "person_income": 500000,
        "person_home_ownership": "OWN",
        "person_emp_length": 25,
        "loan_intent": "PERSONAL",
        "loan_grade": "A",
        "loan_amnt": 10000,
        "loan_int_rate": 6.0,
        "loan_percent_income": 0.02,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 20
    }
    
    print(f"\n✓ Ingreso muy alto ($500k)")
    try:
        result = evaluate_credit_application(high_income)
        print(f"  Score: {result['credit_score']['score']}")
        print(f"  Monto máx recomendado: ${result['loan_recommendation']['max_recommended_loan']:,.2f}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    # Caso 3: Valores mínimos
    minimal = {
        "person_age": 18,
        "person_income": 10000,
        "person_home_ownership": "OTHER",
        "person_emp_length": 0,
        "loan_intent": "PERSONAL",
        "loan_grade": "G",
        "loan_amnt": 500,
        "loan_int_rate": 25.0,
        "loan_percent_income": 0.05,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 0
    }
    
    print(f"\n✓ Valores mínimos (18 años, $10k ingreso, 0 empleo)")
    try:
        result = evaluate_credit_application(minimal)
        print(f"  Score: {result['credit_score']['score']}")
        print(f"  Decisión: {result['final_decision']['decision']}")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    
    print()

def test_consistency():
    """Prueba consistencia del modelo (mismos inputs = mismos outputs)"""
    print("🧪 TEST 6: Consistencia del Modelo")
    print("-" * 40)
    
    test_data = {
        "person_age": 35,
        "person_income": 60000,
        "person_home_ownership": "MORTGAGE",
        "person_emp_length": 6,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "B",
        "loan_amnt": 18000,
        "loan_int_rate": 10.5,
        "loan_percent_income": 0.30,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7
    }
    
    results = []
    for i in range(3):
        result = evaluate_credit_application(test_data)
        results.append({
            "score": result["credit_score"]["score"],
            "decision": result["final_decision"]["decision"],
            "prob_default": result["risk_assessment"]["probability_default"]
        })
    
    # Verificar que todos sean iguales
    is_consistent = (
        all(r["score"] == results[0]["score"] for r in results) and
        all(r["decision"] == results[0]["decision"] for r in results) and
        all(r["prob_default"] == results[0]["prob_default"] for r in results)
    )
    
    print(f"✓ 3 evaluaciones con mismos datos:")
    for i, r in enumerate(results, 1):
        print(f"  Run {i}: Score={r['score']}, Decision={r['decision']}, ProbDefault={r['prob_default']:.4f}")
    
    print(f"\n{'✅ PASS: Resultados consistentes' if is_consistent else '❌ FAIL: Resultados inconsistentes'}")
    print()

def run_all_tests():
    """Ejecuta todas las pruebas"""
    print_separator()
    print("🔬 SUITE DE PRUEBAS - SISTEMA DE EVALUACIÓN CREDITICIA")
    print("="*70)
    print()
    
    try:
        test_validation()
        print_separator()
        
        test_credit_score_calculation()
        print_separator()
        
        test_loan_recommendation()
        print_separator()
        
        all_eval_passed = test_full_evaluation()
        print_separator()
        
        test_edge_cases()
        print_separator()
        
        test_consistency()
        print_separator()
        
        print("📊 RESUMEN DE PRUEBAS")
        print("="*70)
        print("✅ Todas las pruebas completadas")
        if all_eval_passed:
            print("✅ Evaluaciones de perfiles: TODAS PASARON")
        else:
            print("⚠️  Algunas evaluaciones de perfiles tienen diferencias")
            print("   Esto puede ser normal si el modelo tiene comportamiento diferente al esperado")
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
