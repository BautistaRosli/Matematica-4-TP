# A.1 Modelado Cuadrático y Derivación
El modelo cuadrático para ajustar el consumo eléctrico $\hat{y}$ en función de los segundos desde medianoche $x$ es dado por:
$
\begin{equation}
\hat{y}(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2
\end{equation}
$
Los estimadores de mínimos cuadrados ordinarios se obtienen como:
(T es porque es una matriz, de 3x3 (cantidad de parametros) y asi sacar la matriz inversa para poder despejar $\hat{\beta}_0$, $\hat{\beta}_1$ y $\hat{\beta}_2$)
$
\begin{equation}
\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T Y
\end{equation}
$
La derivada primera representa la tasa instantánea de variación del consumo:
$
\begin{equation}
\frac{d\hat{y}}{dx} = \hat{\beta}_1 + 2\hat{\beta}_2 x \quad \left[\text{kWh/s}\right]
\end{equation}
$
Para evaluar la tasa en términos horarios ($\text{kWh/h}$), se aplica la conversión de unidades:
$
\begin{equation}
\left(\frac{d\hat{y}}{dx}\right)_{\text{hora}} = 3600 \cdot (\hat{\beta}_1 + 2\hat{\beta}_2 x) \quad \left[\text{kWh/h}\right]
\end{equation}
$
A diferencia del modelo cuadrático, el modelo lineal $\hat{y}_{\text{lin}}(x) = \hat{\beta}_0 + \hat{\beta}_1 x$ asume una tasa de cambio constante:
$
\begin{equation}
\frac{d\hat{y}_{\text{lin}}}{dx} = \hat{\beta}_1 = \text{cte.}
\end{equation}
$
El punto crítico $x_c$ del modelo cuadrático se halla resolviendo $\frac{d\hat{y}}{dx} = 0$:
$
\begin{equation}
x_c = -\frac{\hat{\beta}_1}{2\hat{\beta}_2}
\end{equation}
$
Su naturaleza se determina mediante la segunda derivada:
$
\begin{equation}
\frac{d^2\hat{y}}{dx^2} = 2\hat{\beta}_2
\end{equation}
$