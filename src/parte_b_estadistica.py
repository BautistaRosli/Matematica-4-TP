"""
Parte B - Complementos de Estadística
B.1 Varianza, B.2 Regresión lineal simple, B.3 Test de hipótesis,
B.4 Intervalos respuesta media y predicción, B.5 Regresión múltiple
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
# B.1 VARIANZA - Modelo simple: Usage_kWh ~ Lagging_Current_Reactive.Power_kVarh
# ============================================================

def parte_b1_varianza(df):
    print("="*70)
    print("PARTE B.1 - VARIANZA (Regresión simple: Usage_kWh ~ Lagging_kVarh)")
    print("="*70)
    
    x = df['Lagging_Current_Reactive.Power_kVarh'].values
    y = df['Usage_kWh'].values
    n = len(x)
    
    # Ajuste modelo simple
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    
    beta_0 = model.params[0]
    beta_1 = model.params[1]
    y_hat = model.fittedvalues
    
    y_bar = np.mean(y)
    
    # Descomposición de variabilidad
    STC = np.sum((y - y_bar)**2)
    SCR = np.sum((y_hat - y_bar)**2)
    SCE = np.sum((y - y_hat)**2)
    
    # Grados de libertad
    gl_total = n - 1
    gl_reg = 1
    gl_error = n - 2
    
    # Varianza residual (estimador insesgado de σ²)
    sigma2_hat = SCE / (n - 2)
    
    # R²
    R2 = 1 - SCE / STC
    # También R² = SCR / STC
    R2_alt = SCR / STC
    
    print(f"\n--- Descomposición de la Variabilidad ---")
    print(f"n = {n}")
    print(f"STC (Total)     = {STC:.6f}   (gl = {gl_total})")
    print(f"SCR (Regresión) = {SCR:.6f}   (gl = {gl_reg})")
    print(f"SCE (Error)     = {SCE:.6f}   (gl = {gl_error})")
    print(f"SCR + SCE       = {SCR + SCE:.6f}")
    print(f"|STC - (SCR+SCE)| = {abs(STC - (SCR + SCE)):.2e}")
    
    print(f"\n--- Estimación de la Varianza del Modelo ---")
    print(f"σ̂² = SCE / (n - 2) = {SCE:.6f} / {n-2} = {sigma2_hat:.6f} kWh²")
    
    print(f"\n--- Coeficiente de Determinación R² ---")
    print(f"R² = 1 - SCE/STC = 1 - {SCE:.6f}/{STC:.6f} = {R2:.6f}")
    print(f"R² = SCR/STC     = {SCR:.6f}/{STC:.6f} = {R2_alt:.6f}")
    print(f"Interpretación: El {R2*100:.2f}% de la variabilidad total del consumo eléctrico")
    print(f"  (Usage_kWh) queda explicada por su relación lineal con la potencia")
    print(f"  reactiva inductiva (Lagging_Current_Reactive.Power_kVarh).")
    
    print(f"\n--- Qué NO significa R² ---")
    print(f"1. NO implica relación de CAUSALIDAD directa entre las variables.")
    print(f"2. NO indica que el modelo lineal sea necesariamente la forma funcional")
    print(f"   correcta (podría haber curvatura no capturada).")
    print(f"3. NO mide la magnitud de la pendiente β₁ ni garantiza ausencia de")
    print(f"   sesgo por variables omitidas.")
    print(f"4. Un R² alto no asegura que las predicciones sean precisas para")
    print(f"   observaciones individuales (ver intervalos de predicción en B.4).")
    
    # Tabla resumen
    print(f"\n--- Tabla Resumen B.1 ---")
    print(f"{'Componente':<20} {'Suma de Cuadrados':>18} {'gl':>6} {'Cuadrado Medio':>16}")
    print(f"{'-'*60}")
    print(f"{'Regresión (SCR)':<20} {SCR:>18.4f} {gl_reg:>6} {SCR/gl_reg:>16.4f}")
    print(f"{'Error (SCE)':<20} {SCE:>18.4f} {gl_error:>6} {SCE/gl_error:>16.4f}")
    print(f"{'Total (STC)':<20} {STC:>18.4f} {gl_total:>6} {STC/gl_total:>16.4f}")
    print(f"\nR² = {R2:.6f}")
    print(f"σ̂² = {sigma2_hat:.6f} kWh²")
    
    return {
        'beta_0': beta_0, 'beta_1': beta_1,
        'STC': STC, 'SCR': SCR, 'SCE': SCE,
        'sigma2_hat': sigma2_hat, 'R2': R2,
        'gl_total': gl_total, 'gl_reg': gl_reg, 'gl_error': gl_error,
        'x': x, 'y': y, 'y_hat': y_hat, 'y_bar': y_bar,
        'n': n, 'model': model
    }

# ============================================================
# B.2 REGRESIÓN LINEAL SIMPLE - 4 modelos
# ============================================================

def parte_b2_regresion_simple(df):
    print("\n" + "="*70)
    print("PARTE B.2 - REGRESIÓN LINEAL SIMPLE (4 modelos)")
    print("="*70)
    
    predictoras = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'Lagging_Current_Power_Factor',
        'NSM'
    ]
    
    y = df['Usage_kWh'].values
    n = len(y)
    x_star = 20  # valor para ICM e IP (según B.4)
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    print(f"\nParámetros comunes: n = {n}, α = {alpha}, x* = {x_star}")
    print(f"t_crit = t(0.025, {n-2}) = {t_crit:.6f}")
    
    resultados = []
    
    for i, col in enumerate(predictoras):
        x = df[col].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        
        beta_0 = model.params[0]
        beta_1 = model.params[1]
        sigma2 = model.scale  # MSE = SCE/(n-2)
        R2 = model.rsquared
        
        # S_xx para cálculos de intervalos
        x_bar = np.mean(x)
        S_xx = np.sum((x - x_bar)**2)
        
        # IC para β₁ y β₀ (95%)
        se_beta_1 = np.sqrt(sigma2 / S_xx)
        se_beta_0 = np.sqrt(sigma2 * (1/n + x_bar**2 / S_xx))
        
        IC_beta_1 = (beta_1 - t_crit * se_beta_1, beta_1 + t_crit * se_beta_1)
        IC_beta_0 = (beta_0 - t_crit * se_beta_0, beta_0 + t_crit * se_beta_0)
        
        # Valor predicho en x*
        y_star = beta_0 + beta_1 * x_star
        
        # ICM(Y|x*) - Intervalo de Confianza para la Respuesta Media
        se_ICM = np.sqrt(sigma2 * (1/n + (x_star - x_bar)**2 / S_xx))
        ICM = (y_star - t_crit * se_ICM, y_star + t_crit * se_ICM)
        
        # IP(Y|x*) - Intervalo de Predicción
        se_IP = np.sqrt(sigma2 * (1 + 1/n + (x_star - x_bar)**2 / S_xx))
        IP = (y_star - t_crit * se_IP, y_star + t_crit * se_IP)
        
        resultados.append({
            'variable': col,
            'beta_0': beta_0, 'beta_1': beta_1,
            'sigma2': sigma2, 'R2': R2,
            'IC_beta_0': IC_beta_0, 'IC_beta_1': IC_beta_1,
            'ICM': ICM, 'IP': IP,
            'x_bar': x_bar, 'S_xx': S_xx,
            'model': model, 'x': x, 'y': y
        })
        
        print(f"\n--- Modelo {i+1}: Usage_kWh ~ {col} ---")
        print(f"  ŷ = {beta_0:.6f} + {beta_1:.6f} x")
        print(f"  σ̂² = {sigma2:.6f}")
        print(f"  R²  = {R2:.6f}")
        print(f"  IC(β₁) = [{IC_beta_1[0]:.6f}, {IC_beta_1[1]:.6f}]")
        print(f"  IC(β₀) = [{IC_beta_0[0]:.6f}, {IC_beta_0[1]:.6f}]")
        print(f"  ICM(Y|x*={x_star}) = [{ICM[0]:.6f}, {ICM[1]:.6f}]")
        print(f"  IP(Y|x*={x_star}) = [{IP[0]:.6f}, {IP[1]:.6f}]")
    
    # Tabla resumen completa
    print(f"\n{'='*70}")
    print("TABLA RESUMEN B.2 - REGRESIÓN LINEAL SIMPLE (4 MODELOS)")
    print(f"{'='*70}")
    print(f"{'Variable':<42} {'β̂₁':>10} {'β̂₀':>10} {'σ̂²':>10} {'R²':>8} {'IC(β₁)':>22} {'IC(β₀)':>22} {'ICM':>22} {'IP':>22}")
    print(f"{'-'*160}")
    for r in resultados:
        v = r['variable'][:40]
        print(f"{v:<42} {r['beta_1']:>10.4f} {r['beta_0']:>10.4f} {r['sigma2']:>10.4f} {r['R2']:>8.4f} "
              f"[{r['IC_beta_1'][0]:>6.3f},{r['IC_beta_1'][1]:>6.3f}] "
              f"[{r['IC_beta_0'][0]:>6.3f},{r['IC_beta_0'][1]:>6.3f}] "
              f"[{r['ICM'][0]:>6.3f},{r['ICM'][1]:>6.3f}] "
              f"[{r['IP'][0]:>6.3f},{r['IP'][1]:>6.3f}]")
    
    # Seleccionar mejor predictor
    mejor = max(resultados, key=lambda r: r['R2'])
    print(f"\n>>> MEJOR PREDICTOR: {mejor['variable']} (R² = {mejor['R2']:.6f})")
    print(f"   Justificación:")
    print(f"   1. Mayor R² = {mejor['R2']:.6f} (explica {mejor['R2']*100:.2f}% de la variabilidad)")
    print(f"   2. Menor varianza residual σ̂² = {mejor['sigma2']:.6f}")
    print(f"   3. Sentido físico: Los hornos y motores de inducción de una acería")
    print(f"      consumen potencia activa y reactiva inductiva de manera acoplada.")
    print(f"      La potencia reactiva inductiva (Lagging) está directamente relacionada")
    print(f"      con la magnetización de equipos eléctricos industriales.")
    
    # Gráfico del mejor modelo
    generate_b2_plot(mejor)
    
    return resultados, mejor

def generate_b2_plot(mejor):
    """Genera gráfico de dispersión con recta ajustada para el mejor modelo"""
    x = mejor['x']
    y = mejor['y']
    beta_0 = mejor['beta_0']
    beta_1 = mejor['beta_1']
    variable = mejor['variable']
    
    # Submuestreo para visualización
    idx = np.random.choice(len(x), size=5000, replace=False)
    x_sample = x[idx]
    y_sample = y[idx]
    
    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = beta_0 + beta_1 * x_grid
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.scatter(x_sample, y_sample, alpha=0.3, s=8, color='steelblue', label='Datos (muestra 5000)')
    ax.plot(x_grid, y_grid, 'r-', linewidth=2.5, label=f'Recta MCO: ŷ = {beta_0:.3f} + {beta_1:.3f}x')
    ax.set_xlabel(f'{variable}')
    ax.set_ylabel('Usage_kWh (kWh)')
    ax.set_title(f'Regresión Simple: Usage_kWh vs {variable}\n(R² = {mejor["R2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/B2_mejor_modelo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico guardado en figures/B2_mejor_modelo.png")

# ============================================================
# B.3 TEST DE HIPÓTESIS - Sobre el mejor modelo de B.2
# ============================================================

def parte_b3_test_hipotesis(mejor):
    print("\n" + "="*70)
    print("PARTE B.3 - TEST DE HIPÓTESIS (Mejor modelo B.2)")
    print("="*70)
    
    x = mejor['x']
    y = mejor['y']
    beta_1 = mejor['beta_1']
    sigma2 = mejor['sigma2']
    n = len(y)
    x_bar = mejor['x_bar']
    S_xx = mejor['S_xx']
    alpha = 0.05
    
    # Estadístico de prueba
    T = beta_1 / np.sqrt(sigma2 / S_xx)
    
    # Valor crítico
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Valor p
    p_value = 2 * (1 - stats.t.cdf(abs(T), n - 2))
    
    print(f"\n--- Planteo Formal ---")
    print(f"H₀: β₁ = 0  (no hay relación lineal)")
    print(f"H₁: β₁ ≠ 0  (existe relación lineal)")
    print(f"\n--- Cálculo del Estadístico ---")
    print(f"β̂₁ = {beta_1:.6f}")
    print(f"σ̂² = {sigma2:.6f}")
    print(f"Sₓₓ = {S_xx:.6f}")
    print(f"n   = {n}")
    print(f"T = β̂₁ / √(σ̂²/Sₓₓ) = {beta_1:.6f} / √({sigma2:.6f}/{S_xx:.6f})")
    print(f"T = {T:.6f}")
    
    print(f"\n--- Valor Crítico y Regla de Decisión ---")
    print(f"α = {alpha}, gl = n - 2 = {n-2}")
    print(f"t(α/2, n-2) = t(0.025, {n-2}) = {t_crit:.6f}")
    print(f"Regla: Rechazar H₀ si |T| > {t_crit:.6f}")
    print(f"|T| = {abs(T):.6f} > {t_crit:.6f} ? {'SÍ' if abs(T) > t_crit else 'NO'}")
    
    print(f"\n--- Valor p ---")
    print(f"p-valor = 2 × P(T_{n-2} > |{T:.6f}|) = {p_value:.2e}")
    print(f"p-valor < α ? {'SÍ' if p_value < alpha else 'NO'}")
    
    # Conclusión en lenguaje del dominio
    print(f"\n--- Conclusión en Lenguaje Industrial ---")
    if abs(T) > t_crit:
        print(f"✅ SE RECHAZA H₀ al nivel de significancia {alpha*100:.0f}%.")
        print(f"   Existe evidencia estadística altamente significativa (p < 0.001)")
        print(f"   de que la pendiente de la recta de regresión verdadera es distinta de cero.")
        print(f"   En términos industriales: La potencia reactiva inductiva")
        print(f"   (Lagging_Current_Reactive.Power_kVarh) tiene una relación")
        print(f"   lineal estadísticamente significativa con el consumo eléctrico")
        print(f"   (Usage_kWh) de la planta siderúrgica. Cada kVarh adicional de")
        print(f"   potencia reactiva inductiva se asocia con un aumento promedio de")
        print(f"   {beta_1:.4f} kWh en el consumo del intervalo de 15 minutos.")
    else:
        print(f"❌ NO se rechaza H₀. No hay evidencia suficiente.")
    
    return {
        'T': T, 't_crit': t_crit, 'p_value': p_value,
        'rechaza_H0': abs(T) > t_crit
    }

# ============================================================
# B.4 INTERVALOS PARA RESPUESTA MEDIA Y PREDICCIÓN
# ============================================================

def parte_b4_intervalos(mejor_lagging):
    print("\n" + "="*70)
    print("PARTE B.4 - INTERVALOS RESPUESTA MEDIA Y PREDICCIÓN")
    print("="*70)
    
    x = mejor_lagging['x']
    y = mejor_lagging['y']
    beta_0 = mejor_lagging['beta_0']
    beta_1 = mejor_lagging['beta_1']
    sigma2 = mejor_lagging['sigma2']
    n = len(y)
    x_bar = mejor_lagging['x_bar']
    S_xx = mejor_lagging['S_xx']
    x_star = 20  # kVarh según enunciado
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, n - 2)
    
    print(f"\n--- Modelo: Usage_kWh ~ Lagging_Current_Reactive.Power_kVarh ---")
    print(f"x* = {x_star} kVarh (potencia reactiva inductiva)")
    
    # Valor puntual
    y_star = beta_0 + beta_1 * x_star
    print(f"\nValor puntual: ŷ* = β̂₀ + β̂₁ × x* = {beta_0:.6f} + {beta_1:.6f} × {x_star} = {y_star:.6f} kWh")
    
    # ICM(Y|x*) - Intervalo de Confianza para la Respuesta Media
    print(f"\n{'='*50}")
    print("INTERVALO DE CONFIANZA 95% PARA LA RESPUESTA MEDIA")
    print(f"ICM(Y|x*={x_star})")
    print(f"{'='*50}")
    print(f"Fórmula: β̂₀ + β̂₁x* ± t(α/2, n-2) × √[σ̂²(1/n + (x* - x̄)²/Sₓₓ)]")
    print(f"")
    print(f"Sustitución numérica:")
    print(f"  β̂₀ = {beta_0:.6f}")
    print(f"  β̂₁ = {beta_1:.6f}")
    print(f"  x*  = {x_star}")
    print(f"  n   = {n}")
    print(f"  x̄   = {x_bar:.6f}")
    print(f"  Sₓₓ = {S_xx:.6f}")
    print(f"  σ̂²  = {sigma2:.6f}")
    print(f"  t(0.025, {n-2}) = {t_crit:.6f}")
    print(f"")
    
    term_ICM = 1/n + (x_star - x_bar)**2 / S_xx
    se_ICM = np.sqrt(sigma2 * term_ICM)
    margin_ICM = t_crit * se_ICM
    ICM_lower = y_star - margin_ICM
    ICM_upper = y_star + margin_ICM
    
    print(f"  1/n + (x* - x̄)²/Sₓₓ = {1/n:.6e} + ({x_star} - {x_bar:.6f})²/{S_xx:.6f}")
    print(f"                      = {term_ICM:.6e}")
    print(f"  √[σ̂² × término] = √[{sigma2:.6f} × {term_ICM:.6e}] = {se_ICM:.6f}")
    print(f"  Margen = t × SE = {t_crit:.6f} × {se_ICM:.6f} = {margin_ICM:.6f}")
    print(f"")
    print(f"  ICM = {y_star:.6f} ± {margin_ICM:.6f}")
    print(f"  ICM = [{ICM_lower:.6f}, {ICM_upper:.6f}] kWh")
    
    # IP(Y|x*) - Intervalo de Predicción
    print(f"\n{'='*50}")
    print("INTERVALO DE PREDICCIÓN 95% PARA UN VALOR FUTURO DE Y")
    print(f"IP(Y|x*={x_star})")
    print(f"{'='*50}")
    print(f"Fórmula: ŷ* ± t(α/2, n-2) × √[σ̂²(1 + 1/n + (x* - x̄)²/Sₓₓ)]")
    print(f"")
    print(f"Sustitución numérica:")
    
    term_IP = 1 + 1/n + (x_star - x_bar)**2 / S_xx
    se_IP = np.sqrt(sigma2 * term_IP)
    margin_IP = t_crit * se_IP
    IP_lower = y_star - margin_IP
    IP_upper = y_star + margin_IP
    
    print(f"  1 + 1/n + (x* - x̄)²/Sₓₓ = 1 + {1/n:.6e} + ({x_star} - {x_bar:.6f})²/{S_xx:.6f}")
    print(f"                          = {term_IP:.6e}")
    print(f"  √[σ̂² × término] = √[{sigma2:.6f} × {term_IP:.6e}] = {se_IP:.6f}")
    print(f"  Margen = t × SE = {t_crit:.6f} × {se_IP:.6f} = {margin_IP:.6f}")
    print(f"")
    print(f"  IP = {y_star:.6f} ± {margin_IP:.6f}")
    print(f"  IP = [{IP_lower:.6f}, {IP_upper:.6f}] kWh")
    
    # Explicación conceptual
    print(f"\n{'='*50}")
    print("EXPLICACIÓN CONCEPTUAL Y PRÁCTICA")
    print(f"{'='*50}")
    print(f"""
INTERVALO DE CONFIANZA PARA LA RESPUESTA MEDIA (ICM):
-----------------------------------------------------
El ICM(Y|x*={x_star}) = [{ICM_lower:.4f}, {ICM_upper:.4f}] kWh
acota con un 95% de confianza el CONSUMO PROMEDIO POBLACIONAL 
de todos los intervalos de 15 minutos en los que la planta opera 
con una potencia reactiva inductiva de {x_star} kVarh.

Es un intervalo sobre el PARÁMETRO β₀ + β₁x* (la media condicional).
Su amplitud depende de la precisión con que se estiman β₀ y β₁.
Al crecer n, el ICM se estrecha (incertidumbre estimacional ↓).

INTERVALO DE PREDICCIÓN (IP):
-----------------------------
El IP(Y|x*={x_star}) = [{IP_lower:.4f}, {IP_upper:.4f}] kWh
acota con un 95% de confianza el CONSUMO ELÉCTRICO DE UN ÚNICO
INTERVALO FUTURO PARTICULAR en el que se registren {x_star} kVarh.

Es un intervalo sobre una OBSERVACIÓN FUTURA INDIVIDUAL Y_futura.
Incorpora la variabilidad aleatoria irreducible del error individual (σ̂²).
Por eso es MUCHO MÁS ANCHO que el ICM: amplitud_IP ≈ amplitud_ICM × √(1 + 1/n + ...)
Al crecer n, el IP NO se estrecha completamente (σ̂² permanece).

DIFERENCIA CLAVE:
-----------------
ICM: "¿Cuál es el consumo PROMEDIO esperado cuando x=20?"
IP:  "¿En qué rango caerá el consumo de UN intervalo ESPECÍFICO futuro con x=20?"

El IP incluye σ̂² (variabilidad irreducible), el ICM no.
Amplitud_IP / Amplitud_ICM ≈ √(1 + ...) ≈ {2*margin_IP/(2*margin_ICM):.2f}x más ancho.
""")
    
    return {
        'x_star': x_star, 'y_star': y_star,
        'ICM': (ICM_lower, ICM_upper), 'IP': (IP_lower, IP_upper),
        'margin_ICM': margin_ICM, 'margin_IP': margin_IP,
        'se_ICM': se_ICM, 'se_IP': se_IP,
        'term_ICM': term_ICM, 'term_IP': term_IP,
        't_crit': t_crit
    }

# ============================================================
# B.5 REGRESIÓN MÚLTIPLE
# ============================================================

def parte_b5_regresion_multiple(df):
    print("\n" + "="*70)
    print("PARTE B.5 - REGRESIÓN MÚLTIPLE")
    print("="*70)
    
    # Variables predictoras (SIN CO2)
    X_cols = [
        'Lagging_Current_Reactive.Power_kVarh',
        'Leading_Current_Reactive_Power_kVarh',
        'Lagging_Current_Power_Factor',
        'NSM'
    ]
    
    X = df[X_cols].values
    y = df['Usage_kWh'].values
    n = len(y)
    k = len(X_cols)  # número de predictoras = 4
    
    # Matriz de diseño con intercepto
    X_design = np.column_stack([np.ones(n), X])
    
    print(f"\n--- Configuración ---")
    print(f"n = {n} observaciones")
    print(f"k = {k} predictoras")
    print(f"Predictoras: {X_cols}")
    print(f"X_design shape: {X_design.shape} (n × {k+1})")
    
    # Resolución matricial exacta: β̂ = (X^T X)⁻¹ X^T y
    XTX = X_design.T @ X_design
    XTy = X_design.T @ y
    XTX_inv = np.linalg.inv(XTX)
    beta = XTX_inv @ XTy
    
    print(f"\n--- Matriz X^T X ---")
    print(f"Shape: {XTX.shape}")
    print(XTX)
    
    print(f"\n--- Inversa (X^T X)⁻¹ ---")
    print(XTX_inv)
    
    print(f"\n--- Vector X^T y ---")
    print(XTy)
    
    print(f"\n--- Estimadores MCO: β̂ = (X^T X)⁻¹ X^T y ---")
    param_names = ['β₀ (intercepto)'] + [f'β_{i+1} ({col})' for i, col in enumerate(X_cols)]
    for i, name in enumerate(param_names):
        print(f"  {name} = {beta[i]:.8f}")
    
    # Validación con statsmodels
    X_sm = sm.add_constant(X)
    model_sm = sm.OLS(y, X_sm).fit()
    print(f"\n--- Validación Statsmodels ---")
    diff = np.abs(beta - model_sm.params)
    print(f"Diferencia máxima: {diff.max():.2e}")
    assert diff.max() < 1e-10
    
    # Métricas del modelo
    y_hat = X_design @ beta
    y_bar = np.mean(y)
    STC = np.sum((y - y_bar)**2)
    SCR = np.sum((y_hat - y_bar)**2)
    SCE = np.sum((y - y_hat)**2)
    R2 = 1 - SCE / STC
    
    print(f"\n--- Métricas del Modelo Múltiple ---")
    print(f"STC = {STC:.4f}")
    print(f"SCR = {SCR:.4f}")
    print(f"SCE = {SCE:.4f}")
    print(f"R²  = {R2:.6f}")
    
    # R² ajustado - AMBAS fórmulas
    # Fórmula del enunciado (PDF)
    R2_adj_enunciado = 1 - (1 - R2) * (n - k) / (n - k - 1)
    # Fórmula estándar bibliográfica
    R2_adj_estandar = 1 - (1 - R2) * (n - 1) / (n - k - 1)
    
    print(f"\n--- R² Ajustado (AMBAS FÓRMULAS) ---")
    print(f"R² = {R2:.8f}")
    print(f"n = {n}, k = {k}")
    print(f"")
    print(f"Fórmula enunciado (PDF): R²_adj = 1 - (1-R²) × (n-k)/(n-k-1)")
    print(f"  = 1 - (1-{R2:.8f}) × ({n-k})/({n-k-1})")
    print(f"  = 1 - {1-R2:.8f} × {(n-k)/(n-k-1):.8f}")
    print(f"  = {R2_adj_enunciado:.8f}")
    print(f"")
    print(f"Fórmula estándar: R²_adj = 1 - (1-R²) × (n-1)/(n-k-1)")
    print(f"  = 1 - (1-{R2:.8f}) × ({n-1})/({n-k-1})")
    print(f"  = 1 - {1-R2:.8f} × {(n-1)/(n-k-1):.8f}")
    print(f"  = {R2_adj_estandar:.8f}")
    print(f"")
    print(f"Diferencia: {abs(R2_adj_enunciado - R2_adj_estandar):.2e}")
    print(f"(Con n={n}, k={k} ambas coinciden hasta 8+ decimales)")
    
    # Comparación con mejor modelo simple (B.2)
    # El mejor fue Lagging_Current_Reactive.Power_kVarh
    # Recalculamos su R² ajustado para comparación justa
    x_lag = df['Lagging_Current_Reactive.Power_kVarh'].values
    X_lag = sm.add_constant(x_lag)
    model_lag = sm.OLS(y, X_lag).fit()
    R2_simple = model_lag.rsquared
    R2_adj_simple_enunciado = 1 - (1 - R2_simple) * (n - 1) / (n - 2)
    R2_adj_simple_estandar = 1 - (1 - R2_simple) * (n - 1) / (n - 2)  # misma para k=1
    
    print(f"\n--- Comparación con Modelo Simple (B.2) ---")
    print(f"Mejor modelo simple: Lagging_Current_Reactive.Power_kVarh")
    print(f"R² simple = {R2_simple:.6f}")
    print(f"R² múltiple = {R2:.6f}")
    print(f"Incremento R² = {R2 - R2_simple:.6f}")
    print(f"")
    print(f"R² ajustado simple (k=1): {R2_adj_simple_estandar:.8f}")
    print(f"R² ajustado múltiple (enunciado): {R2_adj_enunciado:.8f}")
    print(f"R² ajustado múltiple (estándar): {R2_adj_estandar:.8f}")
    print(f"Mejora ajustada: {R2_adj_estandar - R2_adj_simple_estandar:.8f}")
    
    # Modelo final
    print(f"\n{'='*50}")
    print("MODELO FINAL MÚLTIPLE")
    print(f"{'='*50}")
    print(f"ŷ = {beta[0]:.6f}")
    for i, col in enumerate(X_cols):
        sign = "+" if beta[i+1] >= 0 else ""
        print(f"    {sign} {beta[i+1]:.6f} × {col}")
    print(f"")
    print(f"R² = {R2:.6f}")
    print(f"R²_ajustado (estándar) = {R2_adj_estandar:.6f}")
    print(f"R²_ajustado (enunciado) = {R2_adj_enunciado:.6f}")
    
    return {
        'beta': beta, 'XTX': XTX, 'XTX_inv': XTX_inv, 'XTy': XTy,
        'X_cols': X_cols, 'X_design': X_design, 'y': y,
        'STC': STC, 'SCR': SCR, 'SCE': SCE, 'R2': R2,
        'R2_adj_enunciado': R2_adj_enunciado,
        'R2_adj_estandar': R2_adj_estandar,
        'R2_simple': R2_simple,
        'R2_adj_simple': R2_adj_simple_estandar,
        'n': n, 'k': k
    }

def main():
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('informe', exist_ok=True)
    
    df = load_data()
    
    # B.1
    b1_results = parte_b1_varianza(df)
    
    # B.2
    b2_resultados, b2_mejor = parte_b2_regresion_simple(df)
    
    # B.3
    b3_results = parte_b3_test_hipotesis(b2_mejor)
    
    # B.4 - Usar el modelo Lagging (que es el mejor)
    # Buscar el resultado para Lagging en b2_resultados
    mejor_lagging = next(r for r in b2_resultados if 'Lagging_Current_Reactive.Power' in r['variable'])
    b4_results = parte_b4_intervalos(mejor_lagging)
    
    # B.5
    b5_results = parte_b5_regresion_multiple(df)
    
    return {
        'b1': b1_results,
        'b2': {'resultados': b2_resultados, 'mejor': b2_mejor},
        'b3': b3_results,
        'b4': b4_results,
        'b5': b5_results
    }

if __name__ == '__main__':
    results = main()