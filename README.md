# Matemática IV - Trabajo Práctico 2026 (Conjunto A)

Análisis de consumo eléctrico de una planta siderúrgica usando regresión lineal simple y múltiple.

## Estructura del proyecto (tentativa)

```
├── Steel_industry_data.csv          # Dataset (35.040 filas × 11 columnas)
├── TRABAJO.md                       # Enunciado del trabajo práctico
├── AGENTS.md                        # Especificación del agente experto
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore                       # Archivos ignorados por Git
├── src/
│   ├── data_loader.py               # Carga y validación de datos
│   ├── parte_a_analisis.py          # Cálculos Parte A (A.1, A.2, A.3)
│   ├── parte_b_estadistica.py       # Cálculos Parte B (B.1-B.5)
│   └── plots.py                     # Generación de figuras
├── notebooks/
│   └── TP_MateIV_2026_ConjuntoA.ipynb  # Notebook para Google Colab
├── figures/                         # Gráficos generados (PNG/PDF)
└── informe/
    └── resolucion_completa.md       # Informe final con Anexo I
```

## Configuración del entorno virtual

### Opción 1: Usando `venv` (recomendado)

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Usando `conda`

```bash
# Crear entorno conda
conda create -n mate4 python=3.10

# Activar entorno
conda activate mate4

# Instalar dependencias
pip install -r requirements.txt
```

## Verificación de la instalación

```bash
# Verificar que todas las librerías se importan correctamente
python -c "import pandas, numpy, scipy, statsmodels, sklearn, matplotlib, seaborn; print('Todas las librerías importadas correctamente')"

# Verificar versión de Python
python --version
```

## Ejecución en Google Colab

1. Subir el repositorio a GitHub
2. Abrir el notebook en Colab:
   ```
   https://colab.research.google.com/github/<usuario>/<repo>/blob/main/notebooks/TP_MateIV_2026_ConjuntoA.ipynb
   ```
3. La primera celda del notebook instala automáticamente las dependencias con:
   ```python
   !pip install -q -r requirements.txt
   ```

## Validación de datos

Antes de ejecutar cualquier modelo, el script `src/data_loader.py` verifica:
- Dimensiones: `(35040, 11)`
- Ausencia de valores nulos
- Tipos de datos correctos
- Nombres de columnas exactos (incluyendo `Lagging_Current_Reactive.Power_kVarh` con punto)

## Variables principales

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| `Usage_kWh` | Consumo de energía activa (target) | kWh |
| `Lagging_Current_Reactive.Power_kVarh` | Potencia reactiva inductiva | kVarh |
| `Leading_Current_Reactive_Power_kVarh` | Potencia reactiva capacitiva | kVarh |
| `Lagging_Current_Power_Factor` | Factor de potencia inductivo | % |
| `Leading_Current_Power_Factor` | Factor de potencia capacitivo | % |
| `NSM` | Segundos desde medianoche | s |

**Nota:** La columna `CO2(tCO2)` **no debe usarse** como variable predictora en ningún modelo.

## Reproducibilidad

Todas las semillas aleatorias están fijadas a `42` (`np.random.seed(42)`). Los resultados numéricos son 100% reproducibles.