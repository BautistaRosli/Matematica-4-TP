"""
Parte A.2 - Derivadas parciales y ecuaciones normales
Parte A.3 - Descenso del gradiente para regresión múltiple (B.5)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def load_data():
    df = pd.read_csv('Steel_industry_data.csv')
    assert df.shape == (35040, 11)
    assert df.isnull().sum().sum() == 0
    return df

# ============================================================
# PARTE A.2: Derivadas parciales y ecuaciones normales
# ============================================================

def parte_a2_analitico():
    """
    Desarrollo analítico paso a paso de las derivadas parciales
    y ecuaciones normales para regresión lineal simple.
    """
    print("="*70)
    print("PARTE A.2 - DERIVADAS PARCIALES Y ECUACIONES NORMALES")
    print("="*70)
    
    print("""
DESARROLLO ANALÍTICO PASO A PASO
=================================

Dada la función objetivo (suma de cuadrados de los residuos):
f(b₀, b₁) = Σᵢ₌₁ⁿ [yᵢ - (b₀ + b₁xᵢ)]²

1. DERIVADA PARCIAL RESPECTO A b₀:
----------------------------------
∂f/∂b₀ = ∂/∂b₀ Σᵢ [yᵢ - b₀ - b₁xᵢ]²
       = Σᵢ ∂/∂b₀ [yᵢ - b₀ - b₁xᵢ]²          (linearidad de la derivada)
       = Σᵢ 2[yᵢ - b₀ - b₁xᵢ] · ∂/∂b₀ [yᵢ - b₀ - b₁xᵢ]  (regla de la cadena)
       = Σᵢ 2[yᵢ - b₀ - b₁xᵢ] · (-1)
       = -2 Σᵢ (yᵢ - b₀ - b₁xᵢ)

∂f/∂b₀ = -2 Σᵢ (yᵢ - b₀ - b₁xᵢ)

2. DERIVADA PARCIAL RESPECTO A b₁:
----------------------------------
∂f/∂b₁ = ∂/∂b₁ Σᵢ [yᵢ - b₀ - b₁xᵢ]²
       = Σᵢ 2[yᵢ - b₀ - b₁xᵢ] · ∂/∂b₁ [yᵢ - b₀ - b₁xᵢ]
       = Σᵢ 2[yᵢ - b₀ - b₁xᵢ] · (-xᵢ)
       = -2 Σᵢ xᵢ(yᵢ - b₀ - b₁xᵢ)

∂f/∂b₁ = -2 Σᵢ xᵢ(yᵢ - b₀ - b₁xᵢ)

3. ECUACIONES NORMALES (igualando a cero):
------------------------------------------
∂f/∂b₀ = 0  ⟹  -2 Σᵢ (yᵢ - b₀ - b₁xᵢ) = 0  ⟹  Σᵢ yᵢ = n b₀ + b₁ Σᵢ xᵢ
∂f/∂b₁ = 0  ⟹  -2 Σᵢ xᵢ(yᵢ - b₀ - b₁xᵢ) = 0  ⟹  Σᵢ xᵢyᵢ = b₀ Σᵢ xᵢ + b₁ Σᵢ xᵢ²

Sistema de ecuaciones normales:
⎧ n b₀ + b₁ Σ xᵢ     = Σ yᵢ
⎨
⎩ b₀ Σ xᵢ + b₁ Σ xᵢ² = Σ xᵢyᵢ

4. RESOLUCIÓN DEL SISTEMA:
--------------------------
De la primera ecuación:  b₀ = (Σ yᵢ)/n - b₁ (Σ xᵢ)/n = ȳ - b₁ x̄

Sustituyendo en la segunda:
(ȳ - b₁ x̄) Σ xᵢ + b₁ Σ xᵢ² = Σ xᵢyᵢ
ȳ Σ xᵢ - b₁ x̄ Σ xᵢ + b₁ Σ xᵢ² = Σ xᵢyᵢ
b₁ (Σ xᵢ² - x̄ Σ xᵢ) = Σ xᵢyᵢ - ȳ Σ xᵢ

Notando que:
  Σ xᵢ² - x̄ Σ xᵢ = Σ xᵢ² - n x̄² = Σ (xᵢ - x̄)² = Sₓₓ
  Σ xᵢyᵢ - ȳ Σ xᵢ = Σ xᵢyᵢ - n x̄ ȳ = Σ (xᵢ - x̄)(yᵢ - ȳ) = Sₓᵧ

Se obtiene:
β̂₁ = Sₓᵧ / Sₓₓ
β̂₀ = ȳ - β̂₁ x̄

Donde:
  Sₓₓ = Σ (xᵢ - x̄)² = Σ xᵢ² - n x̄²
  Sₓᵧ = Σ (xᵢ - x̄)(yᵢ - ȳ) = Σ xᵢyᵢ - n x̄ ȳ
""")
    
    return None

def parte_a2_numerico(df):
    """
    Cálculo numérico de β̂₀ y β̂₁ usando las fórmulas derivadas
    y contraste con statsmodels/scipy.
    """
    print("\n" + "="*70)
    print("CÁLCULO NUMÉRICO SOBRE LOS DATOS")
    print("="*70)
    
    # Variable respuesta y predictora para A.2
    x = df['Lagging_Current_Reactive.Power_kVarh'].values
    y = df['Usage_kWh'].values
    n = len(x)
    
    # Estadísticos necesarios
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    S_xx = np.sum((x - x_bar)**2)
    S_xy = np.sum((x - x_bar) * (y - y_bar))
    
    print(f"\nEstadísticas básicas (n = {n}):")
    print(f"  x̄ = {x_bar:.6f}")
    print(f"  ȳ = {y_bar:.6f}")
    print(f"  Sₓₓ = Σ(xᵢ - x̄)² = {S_xx:.6f}")
    print(f"  Sₓᵧ = Σ(xᵢ - x̄)(yᵢ - ȳ) = {S_xy:.6f}")
    
    # Estimadores MCO
    beta_1 = S_xy / S_xx
    beta_0 = y_bar - beta_1 * x_bar
    
    print(f"\nEstimadores MCO (fórmulas analíticas):")
    print(f"  β̂₁ = Sₓᵧ/Sₓₓ = {beta_1:.8f}")
    print(f"  β̂₀ = ȳ - β̂₁ x̄ = {beta_0:.8f}")
    print(f"  Recta: ŷ = {beta_0:.6f} + {beta_1:.6f} x")
    
    # Verificación con statsmodels
    X_sm = sm.add_constant(x)
    model_sm = sm.OLS(y, X_sm).fit()
    
    print(f"\nValidación con statsmodels:")
    print(f"  β̂₀ = {model_sm.params[0]:.8f}")
    print(f"  β̂₁ = {model_sm.params[1]:.8f}")
    
    diff_0 = abs(beta_0 - model_sm.params[0])
    diff_1 = abs(beta_1 - model_sm.params[1])
    print(f"\nDiferencias absolutas:")
    print(f"  |β̂₀ - β̂₀_sm| = {diff_0:.2e}")
    print(f"  |β̂₁ - β̂₁_sm| = {diff_1:.2e}")
    
    assert diff_0 < 1e-10 and diff_1 < 1e-10, "Discrepancia con statsmodels"
    print("✓ Coincidencia perfecta (error < 10⁻¹⁰)")
    
    # Verificación con scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print(f"\nValidación con scipy.stats.linregress:")
    print(f"  β̂₀ = {intercept:.8f}")
    print(f"  β̂₁ = {slope:.8f}")
    print(f"  Diferencia β̂₁: {abs(beta_1 - slope):.2e}")
    print(f"  Diferencia β̂₀: {abs(beta_0 - intercept):.2e}")
    
    # Tabla comparativa
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA DE VALIDACIÓN")
    print(f"{'='*70}")
    print(f"{'Método':<30} {'β̂₀':>15} {'β̂₁':>15}")
    print(f"{'-'*60}")
    print(f"{'Fórmulas analíticas (NumPy)':<30} {beta_0:>15.8f} {beta_1:>15.8f}")
    print(f"{'statsmodels OLS':<30} {model_sm.params[0]:>15.8f} {model_sm.params[1]:>15.8f}")
    print(f"{'scipy.stats.linregress':<30} {intercept:>15.8f} {slope:>15.8f}")
    print(f"{'Diferencia máx. (analítico-sm)':<30} {diff_0:>15.2e} {diff_1:>15.2e}")
    
    return beta_0, beta_1, x, y, x_bar, y_bar, S_xx, S_xy

def parte_a2_suma_cuadrados(df, beta_0, beta_1, x, y):
    """Calcula SCE, SCR, STC para verificación posterior en B.1"""
    y_hat = beta_0 + beta_1 * x
    y_bar = np.mean(y)
    
    SCE = np.sum((y - y_hat)**2)
    SCR = np.sum((y_hat - y_bar)**2)
    STC = np.sum((y - y_bar)**2)
    
    print(f"\nVerificación descomposición variabilidad (para B.1):")
    print(f"  STC = {STC:.6f}")
    print(f"  SCR = {SCR:.6f}")
    print(f"  SCE = {SCE:.6f}")
    print(f"  SCR + SCE = {SCR + SCE:.6f}")
    print(f"  |STC - (SCR+SCE)| = {abs(STC - (SCR + SCE)):.2e}")
    
    return SCE, SCR, STC

# ============================================================
# PARTE A.3: Descenso del gradiente para regresión múltiple (B.5)
# ============================================================

def prepare_multiple_regression_data(df):
    """Prepara matriz de diseño X y vector Y para regresión múltiple B.5"""
    # Variables predictoras (SIN CO2)
    X_cols = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'Lagging_Current_Power_Factor',
        'NSM'
    ]
    
    X = df[X_cols].values
    y = df['Usage_kWh'].values
    
    # Estandarización (z-score) para mejor condicionamiento
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Añadir columna de unos para intercepto
    X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    
    # También versión sin escalar para comparación
    X_orig = np.column_stack([np.ones(len(X)), X])
    
    return X_scaled, y, X_orig, X_mean, X_std, X_cols

def cost_function(X, y, beta):
    """Función de costo: MSE (1/n) * ||y - Xβ||²"""
    n = len(y)
    residuals = y - X @ beta
    return (residuals.T @ residuals) / n

def gradient(X, y, beta):
    """Gradiente de la función de costo: -(2/n) X^T (y - Xβ)"""
    n = len(y)
    return -(2/n) * X.T @ (y - X @ beta)

def gradient_descent(X, y, alpha, beta_init, max_iter=10000, tol=1e-8):
    """
    Descenso del gradiente con criterio de parada por cambio relativo en parámetros.
    Retorna: beta_final, history_cost, history_beta, n_iter, converged
    """
    beta = beta_init.copy()
    history_cost = []
    history_beta = []
    
    for k in range(max_iter):
        cost = cost_function(X, y, beta)
        history_cost.append(cost)
        history_beta.append(beta.copy())
        
        grad = gradient(X, y, beta)
        beta_new = beta - alpha * grad
        
        # Criterio de parada: cambio relativo en norma de parámetros
        change = np.linalg.norm(beta_new - beta) / (np.linalg.norm(beta) + 1e-12)
        beta = beta_new
        
        if change < tol:
            return beta, history_cost, history_beta, k+1, True
    
    return beta, history_cost, history_beta, max_iter, False

def parte_a3_gradient_descent(df):
    """
    Ejecuta descenso del gradiente con tres tasas de aprendizaje
    para el modelo múltiple de B.5.
    """
    print("\n" + "="*70)
    print("PARTE A.3 - DESCENSO DEL GRADIENTE (REGRESIÓN MÚLTIPLE B.5)")
    print("="*70)
    
    # Preparar datos
    X_scaled, y, X_orig, X_mean, X_std, X_cols = prepare_multiple_regression_data(df)
    n, p = X_scaled.shape  # p = 5 (intercepto + 4 predictoras)
    
    print(f"\nDatos preparados:")
    print(f"  n = {n}, p = {p} (incluye intercepto)")
    print(f"  Predictoras: {X_cols}")
    print(f"  Escalado: z-score (media=0, std=1)")
    
    # Solución exacta MCO en espacio escalado
    beta_ols_scaled = np.linalg.inv(X_scaled.T @ X_scaled) @ X_scaled.T @ y
    cost_ols = cost_function(X_scaled, y, beta_ols_scaled)
    
    print(f"\nSolución exacta MCO (espacio escalado):")
    print(f"  β̂ = {beta_ols_scaled}")
    print(f"  Costo MSE = {cost_ols:.8f}")
    
    # Transformar coeficientes al espacio original
    # β₀_orig = β₀_scaled - Σ(βⱼ_scaled * x̄ⱼ / sⱼ)
    # βⱼ_orig = βⱼ_scaled / sⱼ  para j=1..4
    beta_ols_orig = np.zeros(p)
    beta_ols_orig[0] = beta_ols_scaled[0] - np.sum(beta_ols_scaled[1:] * X_mean / X_std)
    beta_ols_orig[1:] = beta_ols_scaled[1:] / X_std
    
    print(f"\nCoeficientes MCO en espacio original:")
    print(f"  β̂₀ = {beta_ols_orig[0]:.8f}")
    for j, col in enumerate(X_cols):
        print(f"  β̂_{j+1} ({col}) = {beta_ols_orig[j+1]:.8f}")
    
    # Validación con statsmodels (espacio original)
    X_sm = sm.add_constant(df[X_cols].values)
    model_sm = sm.OLS(y, X_sm).fit()
    print(f"\nValidación statsmodels (espacio original):")
    print(f"  β̂ = {model_sm.params}")
    diff = np.abs(beta_ols_orig - model_sm.params)
    print(f"  Diferencia máx = {diff.max():.2e}")
    assert diff.max() < 1e-10
    
    # ---------------------------------------------------
    # ANÁLISIS DE CONDICIONAMIENTO Y TASAS DE APRENDIZAJE
    # ---------------------------------------------------
    print(f"\n{'='*70}")
    print("ANÁLISIS NUMÉRICO: CONDICIONAMIENTO Y TASAS DE APRENDIZAJE")
    print(f"{'='*70}")
    
    # Matriz Hessiana = (2/n) X^T X (en espacio escalado)
    H = (2/n) * X_scaled.T @ X_scaled
    eigenvals = np.linalg.eigvals(H)
    eigenvals = np.real(eigenvals)  # Son reales (H simétrica), descartar parte imag. nula
    lambda_max = eigenvals.max()
    lambda_min = eigenvals.min()
    cond_num = lambda_max / lambda_min
    
    print(f"\nAutovalores de la Hessiana (2/n X^T X):")
    print(f"  λ_max = {lambda_max:.6f}")
    print(f"  λ_min = {lambda_min:.6f}")
    print(f"  Número de condición κ = λ_max/λ_min = {cond_num:.2f}")
    
    # Cota de estabilidad: α < 2/λ_max
    alpha_max_stable = 2 / lambda_max
    print(f"\nCota de estabilidad teórica: α < 2/λ_max = {alpha_max_stable:.6f}")
    
    # Tres tasas de aprendizaje
    alpha_small = alpha_max_stable * 0.01      # Muy pequeña
    alpha_good = alpha_max_stable * 0.5        # Adecuada (factor de seguridad 0.5)
    alpha_large = alpha_max_stable * 1.5       # Divergente
    
    print(f"\nTasas de aprendizaje seleccionadas:")
    print(f"  α_pequeña   = {alpha_small:.6f}  (0.01 × α_max)")
    print(f"  α_adecuada  = {alpha_good:.6f}   (0.5 × α_max)")
    print(f"  α_divergente = {alpha_large:.6f}   (1.5 × α_max) → DIVERGE")
    
    # Inicialización en ceros
    beta_init = np.zeros(p)
    
    # Ejecutar tres experimentos
    print(f"\n{'='*70}")
    print("EJECUCIÓN DE LOS TRES EXPERIMENTOS")
    print(f"{'='*70}")
    
    results = {}
    
    for name, alpha in [("pequeña", alpha_small), ("adecuada", alpha_good), ("divergente", alpha_large)]:
        print(f"\n--- α = {alpha:.6f} ({name}) ---")
        beta_final, history_cost, history_beta, n_iter, converged = gradient_descent(
            X_scaled, y, alpha, beta_init, max_iter=50000, tol=1e-8
        )
        
        # Transformar a espacio original para comparación
        beta_final_real = np.real(beta_final)
        beta_final_orig = np.zeros(p)
        beta_final_orig[0] = beta_final_real[0] - np.sum(beta_final_real[1:] * X_mean / X_std)
        beta_final_orig[1:] = beta_final_real[1:] / X_std
        
        diff_params = np.linalg.norm(beta_final_orig - beta_ols_orig)
        diff_cost = abs(history_cost[-1] - cost_ols)
        
        print(f"  Iteraciones: {n_iter}")
        print(f"  Convergió: {'SÍ' if converged else 'NO'}")
        print(f"  Costo final: {history_cost[-1]:.8f} (MCO: {cost_ols:.8f})")
        print(f"  Diff costo: {diff_cost:.2e}")
        print(f"  Diff params (norma): {diff_params:.2e}")
        
        if name == "divergente" and not converged:
            print(f"  ✓ Comportamiento esperado: DIVERGENCIA (costo crece)")
        
        results[name] = {
            'alpha': alpha,
            'beta_final': beta_final,
            'beta_final_orig': beta_final_orig,
            'history_cost': history_cost,
            'history_beta': history_beta,
            'n_iter': n_iter,
            'converged': converged,
            'diff_params': diff_params,
            'diff_cost': diff_cost
        }
    
    # Gráfico conjunto de convergencia
    generate_gd_plot(results, cost_ols)
    
    # Tabla comparativa final
    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA: GRADIENTE DESCENDIENTE vs MCO EXACTO")
    print(f"{'='*70}")
    print(f"{'Parámetro':<30} {'MCO Exacto':>15} {'GD (α adecuada)':>18} {'Diff':>12}")
    print(f"{'-'*75}")
    param_names = ['β₀ (intercepto)'] + [f'β_{j+1} ({col})' for j, col in enumerate(X_cols)]
    for i, name in enumerate(param_names):
        print(f"{name:<30} {beta_ols_orig[i]:>15.8f} {results['adecuada']['beta_final_orig'][i]:>18.8f} {abs(beta_ols_orig[i] - results['adecuada']['beta_final_orig'][i]):>12.2e}")
    
    return results, beta_ols_orig, X_cols, cost_ols

def generate_gd_plot(results, cost_ols):
    """Genera gráfico de convergencia para las tres tasas"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'pequeña': 'blue', 'adecuada': 'green', 'divergente': 'red'}
    labels = {'pequeña': 'α pequeña (lento)', 'adecuada': 'α adecuada (rápido)', 'divergente': 'α grande (diverge)'}
    
    # Panel 1: Escala lineal (zoom en convergentes)
    ax = axes[0]
    for name in ['pequeña', 'adecuada']:
        history = results[name]['history_cost']
        ax.plot(range(len(history)), history, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(cost_ols, color='black', linestyle='--', linewidth=1, label='MCO exacto')
    ax.set_xlabel('Iteración k')
    ax.set_ylabel('Costo MSE')
    ax.set_title('Convergencia: α pequeña vs α adecuada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=cost_ols * 0.99, top=cost_ols * 2)
    
    # Panel 2: Escala logarítmica (las tres)
    ax = axes[1]
    for name in ['pequeña', 'adecuada', 'divergente']:
        history = results[name]['history_cost']
        # Evitar log(0) o valores negativos por redondeo
        history_safe = np.maximum(np.array(history), 1e-16)
        ax.semilogy(range(len(history)), history_safe, color=colors[name], linewidth=2, label=labels[name])
    ax.axhline(cost_ols, color='black', linestyle='--', linewidth=1, label='MCO exacto')
    ax.set_xlabel('Iteración k')
    ax.set_ylabel('Costo MSE (escala log)')
    ax.set_title('Tres regímenes: convergencia lenta, rápida y divergencia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/A3_gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nGráfico guardado en figures/A3_gradient_descent.png")

def main():
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('informe', exist_ok=True)
    
    # Cargar datos
    df = load_data()
    
    # ---- PARTE A.2 ----
    parte_a2_analitico()
    beta_0, beta_1, x, y, x_bar, y_bar, S_xx, S_xy = parte_a2_numerico(df)
    parte_a2_suma_cuadrados(df, beta_0, beta_1, x, y)
    
    # ---- PARTE A.3 ----
    results, beta_ols_orig, X_cols, cost_ols = parte_a3_gradient_descent(df)
    
    # Retornar resultados para el informe
    return {
        'a2': {
            'beta_0': beta_0, 'beta_1': beta_1,
            'x_bar': x_bar, 'y_bar': y_bar,
            'S_xx': S_xx, 'S_xy': S_xy,
            'x': x, 'y': y
        },
        'a3': {
            'results': results,
            'beta_ols_orig': beta_ols_orig,
            'X_cols': X_cols,
            'cost_ols': cost_ols
        }
    }

if __name__ == '__main__':
    results = main()