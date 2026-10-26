# A.1 Modelado Cuadrático y Derivación

El modelo cuadrático para ajustar el consumo eléctrico \(\hat{y}\) en función de los segundos desde medianoche \(x\) es dado por:

$$
\hat{y}(x) = \hat{\beta}_0 + \hat{\beta}_1 x + \hat{\beta}_2 x^2
$$

Los estimadores de mínimos cuadrados ordinarios se obtienen como:

$$
\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T Y
$$

\(X\) es la matriz de diseño de dimensión \(n \times 3\), correspondiente a los tres parámetros del modelo. La expresión permite calcular los estimadores \(\hat{\beta}_0\), \(\hat{\beta}_1\) y \(\hat{\beta}_2\).

La derivada primera representa la tasa instantánea de variación del consumo:

$$
\frac{d\hat{y}}{dx} = \hat{\beta}_1 + 2\hat{\beta}_2 x
\quad [\text{kWh/s}]
$$

Para evaluar la tasa en términos horarios (\(\text{kWh/h}\)), se aplica la conversión de unidades:

$$
\left(\frac{d\hat{y}}{dx}\right)_{\text{hora}}
= 3600 \cdot (\hat{\beta}_1 + 2\hat{\beta}_2 x)
\quad [\text{kWh/h}]
$$

A diferencia del modelo cuadrático, el modelo lineal \(\hat{y}_{\text{lin}}(x) = \hat{\beta}_0 + \hat{\beta}_1 x\) asume una tasa de cambio constante:

$$
\frac{d\hat{y}_{\text{lin}}}{dx}
= \hat{\beta}_1
= \text{cte.}
$$

El punto crítico \(x_c\) del modelo cuadrático se halla resolviendo \(\frac{d\hat{y}}{dx} = 0\):

$$
x_c = -\frac{\hat{\beta}_1}{2\hat{\beta}_2}
$$

Su naturaleza se determina mediante la segunda derivada:

$$
\frac{d^2\hat{y}}{dx^2} = 2\hat{\beta}_2
$$