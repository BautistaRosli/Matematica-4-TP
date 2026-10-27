**# 2. Punto A.2 — Derivadas Parciales y Ecuaciones Normales**

**## 2.1 Planteo de la Función Objetivo**

Para la regresión lineal simple de $y$ (`Usage_kWh`) sobre $x$ (`Lagging_Current_Reactive.Power_kVarh`), la función de suma de residuos al cuadrado (SCE) a minimizar en función de los parámetros $b_0$ y $b_1$ es:

$$
\begin{equation}
f(b_0, b_1) = \sum_{i=1}^{n} \left[ y_i - (b_0 + b_1 x_i) \right]^2
\end{equation}
$$

**## 2.2 Desarrollo algebraico de las derivadas parciales**

**#### A. Derivada Parcial respecto a $b_0$:**

Aplicando la propiedad de linealidad del operador sumatoria y la regla de la cadena:

$$
\begin{equation}
\frac{\partial f}{\partial b_0} = \frac{\partial}{\partial b_0} \sum_{i=1}^{n} \left[ y_i - b_0 - b_1 x_i \right]^2
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_0} = \sum_{i=1}^{n} \frac{\partial}{\partial b_0} \left[ y_i - b_0 - b_1 x_i \right]^2
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_0} = \sum_{i=1}^{n} 2 \left[ y_i - b_0 - b_1 x_i \right] \cdot \frac{\partial}{\partial b_0} \left( y_i - b_0 - b_1 x_i \right)
\end{equation}
$$

Puesto que $\frac{\partial}{\partial b_0}(y_i - b_0 - b_1 x_i) = -1$:

$$
\begin{equation}
\frac{\partial f}{\partial b_0} = \sum_{i=1}^{n} 2 \left[ y_i - b_0 - b_1 x_i \right] \cdot (-1)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_0} = -2 \sum_{i=1}^{n} (y_i - b_0 - b_1 x_i)
\end{equation}
$$

---

**#### B. Derivada Parcial respecto a $b_1$:**

De manera análoga, aplicando la regla de la cadena:

$$
\begin{equation}
\frac{\partial f}{\partial b_1} = \frac{\partial}{\partial b_1} \sum_{i=1}^{n} \left[ y_i - b_0 - b_1 x_i \right]^2
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_1} = \sum_{i=1}^{n} \frac{\partial}{\partial b_1} \left[ y_i - b_0 - b_1 x_i \right]^2
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_1} = \sum_{i=1}^{n} 2 \left[ y_i - b_0 - b_1 x_i \right] \cdot \frac{\partial}{\partial b_1} \left( y_i - b_0 - b_1 x_i \right)
\end{equation}
$$

Como $\frac{\partial}{\partial b_1}(y_i - b_0 - b_1 x_i) = -x_i$:

$$
\begin{equation}
\frac{\partial f}{\partial b_1} = \sum_{i=1}^{n} 2 \left[ y_i - b_0 - b_1 x_i \right] \cdot (-x_i)
\end{equation}
$$

$$
\begin{equation}
\frac{\partial f}{\partial b_1} = -2 \sum_{i=1}^{n} x_i (y_i - b_0 - b_1 x_i)
\end{equation}
$$

---

**### 2.3 Planteo del Sistema de Ecuaciones Normales**

Igualando ambas derivadas parciales a cero para llegar al punto crítico minimizante:

**1. Primera ecuación:**

$$
\begin{equation}
-2 \sum_{i=1}^{n} (y_i - b_0 - b_1 x_i) = 0 \implies \sum_{i=1}^{n} y_i - \sum_{i=1}^{n} b_0 - b_1 \sum_{i=1}^{n} x_i = 0
\end{equation}
$$

Como $\sum_{i=1}^{n} b_0 = n b_0$:

$$
\begin{equation}
n b_0 + b_1 \sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i \quad \text{--- (1ª Ecuación Normal)}
\end{equation}
$$

**2. Segunda ecuación:**

$$
\begin{equation}
-2 \sum_{i=1}^{n} x_i (y_i - b_0 - b_1 x_i) = 0 \implies \sum_{i=1}^{n} x_i y_i - b_0 \sum_{i=1}^{n} x_i - b_1 \sum_{i=1}^{n} x_i^2 = 0
\end{equation}
$$

$$
\begin{equation}
b_0 \sum_{i=1}^{n} x_i + b_1 \sum_{i=1}^{n} x_i^2 = \sum_{i=1}^{n} x_i y_i \quad \text{--- (2ª Ecuación Normal)}
\end{equation}
$$

Organizado en formato de sistema $2 \times 2$:

$$
\begin{equation}
\begin{cases}
n b_0 + b_1 \displaystyle\sum_{i=1}^{n} x_i = \displaystyle\sum_{i=1}^{n} y_i \\[12pt]
b_0 \displaystyle\sum_{i=1}^{n} x_i + b_1 \displaystyle\sum_{i=1}^{n} x_i^2 = \displaystyle\sum_{i=1}^{n} x_i y_i
\end{cases}
\end{equation}
$$

---

**### 2.4 Despeje Analítico Completo de los Estimadores MCO**

**1. Despeje de $b_0$ de la 1ª Ecuación Normal:**

$$
\begin{equation}
b_0 = \frac{1}{n} \sum_{i=1}^{n} y_i - b_1 \frac{1}{n} \sum_{i=1}^{n} x_i = \bar{y} - b_1 \bar{x}
\end{equation}
$$

**2. Sustitución de $b_0$ en la 2ª Ecuación Normal:**

$$
\begin{equation}
\left(\bar{y} - b_1 \bar{x}\right) \sum_{i=1}^{n} x_i + b_1 \sum_{i=1}^{n} x_i^2 = \sum_{i=1}^{n} x_i y_i
\end{equation}
$$

$$
\begin{equation}
\bar{y} \sum_{i=1}^{n} x_i - b_1 \bar{x} \sum_{i=1}^{n} x_i + b_1 \sum_{i=1}^{n} x_i^2 = \sum_{i=1}^{n} x_i y_i
\end{equation}
$$

$$
\begin{equation}
b_1 \left( \sum_{i=1}^{n} x_i^2 - \bar{x} \sum_{i=1}^{n} x_i \right) = \sum_{i=1}^{n} x_i y_i - \bar{y} \sum_{i=1}^{n} x_i
\end{equation}
$$

**3. Uso de las relaciones muestrales centradas en la media:**

Recordando que $\sum x_i = n \bar{x}$ y $\sum y_i = n \bar{y}$:

$$
\begin{equation}
\sum_{i=1}^{n} x_i^2 - \bar{x} (n \bar{x}) = \sum_{i=1}^{n} x_i^2 - n \bar{x}^2 = \sum_{i=1}^{n} (x_i - \bar{x})^2 = S_{xx}
\end{equation}
$$

$$
\begin{equation}
\sum_{i=1}^{n} x_i y_i - \bar{y} (n \bar{x}) = \sum_{i=1}^{n} x_i y_i - n \bar{x} \bar{y} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) = S_{xy}
\end{equation}
$$

**4. Expresiones finales de los estimadores MCO:**

$$
\begin{equation}
\hat{\beta}\_1 = \frac{S\_{xy}}{S\_{xx}} = \frac{\displaystyle\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\displaystyle\sum_{i=1}^{n} (x_i - \bar{x})^2}
\end{equation}
$$

$$
\begin{equation}
\hat{\beta}\_0 = \bar{y} - \hat{\beta}\_1 \bar{x}
\end{equation}
$$