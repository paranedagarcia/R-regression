# R-regression
Análisis de regresion en R. Parte del material de curso universitario Analítica de datos.

# ¿Qué es regresión?

Es un tipo de análisis estadístico para establecer la posible relación entre una o más variables de un conjunto de datos observados. Se intenta determinar la relación entre una variable de interés (dependiente) respecto del valor de una o más variables predictoras. Todas las variables involucradas (dependiente y las predictoras utilizadas) deben ser numéricas para este tipo de análisis.

Este análisis también permite establecer el grado de relación entre una o más variables indicando cuál de las variables predictoras tiene mayor o menor incidencia en el valor de la variable dependiente. Esto se logra mediante el **coeficiente de correlación**, donde 1.0 indica correlación completa y 0 nula correlación.

> **NOTA**: la correlación no indica causalidad.

El modelo de regresión lineal es una manera simple de establecer predicciones entre variables, con una fórmula matemática simple. Se utiliza en todo ámbito, desde las ciencias biológicas a los negocios.

El modelo de regresión se clasifica en los siguientes tipos:

-   **Regresión lineal simple**: relación entre una variable numérica dependiente y una variable numérica predictora.

-   **Regresión lineal múltiple**: relación entre una variable dependiente numérica y dos o más variables numéricas predictoras.

-   **Regresión logística**: relación entre una variable cualitativa dependiente (dicotómica) y un conjunto de variables numéricas.

# Regresión lineal simple

La regresión lineal presenta la relación entre variables numéricas, más expresamente entre una variable dependiente (y) y otra variable predictora (x). Intenta por tanto, predecir el valor de una variable cuantitativa en relación a otra.

La regresión lineal simple asume una solución de ecuación de la recta del tipo:

$$y = \beta_{0} + \beta_{1} * x + \epsilon$$

donde $\beta_{0}$ corresponde al valor de intersección en el eje de la ordenada de la variable dependiente (y), y $\beta_{1}$ al valor de la pendiente de la línea recta, ambos son valores constantes. Y $\epsilon$ es un error estimado dado por la diferencia entre la observación y el valor entregado por el modelo.

## Cuantificación del error

Cuando se construye un modelo de regresión, es necesario evaluar el rendimiento del modelo predictivo. En otras palabras, hay que evaluar la eficacia del modelo para predecir el resultado de unos nuevos datos de prueba que no se han utilizado para construir el modelo.

Para evaluar el rendimiento del modelo de regresión predictiva se suelen utilizar dos métricas importantes:

**Raíz del error cuadrático medio**

Esta médida (Root Mean Squared Error **RMSE**), mide el error de predicción del modelo. Y que corresponde a la diferencia media entre los valores conocidos observados del resultado y el valor predicho por el modelo. El RMSE se calcula como la raíz media((predicho - observado)\^2) .

> **Nota**: Cuanto menor sea el RMSE, mejor será el modelo.

$$RMSE =\sqrt[]{\dfrac{\sum\limits_{i=1}^{N} (yp -yr)^2}{N}}$$

Este valor es una medida de precisión del modelo aplicado al conjunto de datos, y sirve para comparar con otros modelos aplicados a ese mismo conjunto de datos. Al ser al cuadrado, los errores de mayor valor inciden mayormente en el error, por ello es sensible a los valores atípicos ("outliers").

**R-cuadrado**

El R-cuadrado (R-square) o coeficiente de determinación, es una medida que indica qué tan cerca se encuentran los datos de la línea de regresión. Se establece entre 0 y 100%.

> **Nota:** Cuanto mayor sea el R2, mejor será el ajuste el modelo a los datos.

R-cuadrado = variación explicada / variación total

100% implica que los valores ajustados son iguales a los valores observados y que por tanto todos los puntos están entonces sobre la recta.

## Función lm

La función "Linear Model" lm() es la función principal dentro de R para el cálculo de ajuste de un modelo lineal simple.

    lm(formula, data, subset, weights, na.action, method = "qr")

Donde (información más detallada se encuentra en la ayuda de R):

-   formula: corresponde a las variables y covariables utilizadas; y \~ x1 + x2 +xn indica que y es la variable respuesta y x1, x2 son las covariables.

-   data: establece el conjunto de datos desde donde se obtienen las variables.

-   na.action: qué hacer frente a valores nulos, por defecto se utiliza "**na.omit**"

-   method: método utilizado para el ajuste. Por defecto es "**qr**" (raíz cuadrada)

Tomando los datos del set de datos "cars" (disponible en forma predeterminada en R), se evidencia una "relación relativa" entre la velocidad del automóvil y la distancia necesaria para el frenado.


```{r libraries, message=FALSE, warning=FALSE}
# librerias a cargar
library(mlbench)  # dataset destinados a pruebas de machine learning
library(tidyverse)# 
library(corrplot) # 
library(plotly)   #
library(caret)    # creación de entrenamiento de clasificación y regresión
library(caTools)  # 
library(reshape2) # 
```

```{r echo=FALSE}
theme_set(theme_bw())
theme_update(panel.grid.major=element_blank(), 
             panel.grid.minor= element_blank()
#             axis.title = element_text(color = "white"),
#             axis.text.x = element_text(color = "white"),
#             axis.text.y = element_text(color = "white"),
#             axis.line = element_line(colour = "white"),
#             axis.ticks = element_line(colour = "white"),
#             panel.background = element_rect(fill = "#474C4E")
             )
```

```{r}
# velocidad y distancia en set de datos cars.
cars %>%
  ggplot(aes(x = speed, y = dist)) +
  geom_point()
```
### Análisis de normalidad de las variables

```{r}
par(mfrow = c(1, 2)) # determinar que se grafique en una fila con 2 columnas
hist(cars$speed, breaks = 10, main = "", xlab = "Speed")
hist(cars$dist, breaks = 10, main = "", xlab = "Distance", border = "blue")
```


### Regresión lineal

Estableciendo la regresión lineal mediante el uso de la función modelo lineal (lm()), el resultado es un objeto con los coeficientes que corresponden a:

-   Intercept: el valor que cruza el eje de ordenadas de la variable dependiente (y).

-   El valor de la pendiente para la variable predictora (x).

```{r}
regresion <- lm(cars$dist ~ cars$speed)
regresion
```

¿Qué tan bien se ajusta el modelo?

Para determinar la bondad del ajuste ver el $R^2$:
```{r}
summary(regresion)
```
- **R-squared** con un valor de 0.8345 indica que el ajuste es relativaente bueno, más cerca de uno. La ventaja es que es independiente de la escala en la que se mida la variable respuesta, por lo que su interpretación es más sencilla.

- **Pr(>|t|)**: corresponde al valor-p para las pruebas de hipótesis de que los coeficientes son ceros. Si esta valor-p es menor que el nivel de significancia (5%), entonces se rechaza la hipótesis nula y por lo tanto, existe significancia en la regresión. Las pruebas de hipótesis de que los coeficientes de las variables regresoras son cero, son muy importantes; debido a que, si se prueba que el coeficiente es cero, entonces la variable regresora no tendría ningún aporte al modelo.

- **Residual standard error**: es la estimación de la desviación estándar. Este valor da una idea de cuán lejos están los valores del modelo ajustado a los valores observados de la variable respuesta.

- **Multiple R-squared**: este valor expresa la proporción de la variación explicada por el modelo; es decir, por las variables explicativas. En este caso, aproximadamente el 65% de la variación en la variable respuesta puede ser explicada por el modelo; es decir, por las variables regresoras (x1 y x2).

- **p-value**: Si es bajo (< 0.05) significa que hay al menos un predictor que tiene relación con la respuesta.

coef(regresion)[[2]] extrae los coeficientes del objeto **regresion** que es el resultado del modelo obtenido con lm().

```{r}
# grafica de los datos más el modelo de regresión
cars %>%
  ggplot(aes(x = speed, y = dist)) +
  geom_point() +
  geom_abline(slope = coef(regresion)[[2]], # valor de la pendiente
              intercept = coef(regresion)[[1]], # valor de intercepción en y
              color = 'red') +
  theme_bw()
# alternativa: geom_smooth retorna la misma linea basdo en el modelo "lm"
# geom_smooth(method = "lm", se = TRUE)
```

Agregamos los valores predichos según el cálculo de regresión, para visualizar en un gráfico la distancia a los valores reales.

```{r}
# 
cars$predicciones <- predict(regresion)

ggplot(cars, aes(x=speed, y=dist)) +
  #geom_smooth(method="lm", se=FALSE, color="lightgrey") +
  geom_abline(slope = coef(regresion)[[2]], 
              intercept = coef(regresion)[[1]], 
              color = 'red') +
  geom_segment(aes(xend=speed, yend=predicciones), col='red', lty='dashed') +
  geom_point() +
  geom_point(aes(y=predicciones), col='red') +
  theme_bw()
```

```{r}
par(mfrow = c(2,2)) # Indicamos que queremos 2 gráficos por pantalla (1 fila y 2 columnas)
plot(regresion) # Pintamos los graficos de residuos
```

## Dataset Orange

Es un set de datos de cultivo de naranjas, que registra la vida en días de 5 árboles y los diámetros de sus frutos expresados en milimetros (mm).

```{r}
data("Orange")
```

```{r}
Orange %>%
  ggplot(aes(x = age, y = circumference)) +
  geom_point()
```

### Cálculo de regresión

Utilizando el comando lm (linear models) donde sus argumentos contemplan y \~ x, donde y es la variable dependiente y x la variable independiente. El segundo argumento corresponde a los datos, almacenados en una variable objeto. El objeto resultante es el modelo de regresion con toda la información resultante.

```{r}
# ajuste lineal
regresion = lm(circumference ~ age, data = Orange)
regresion
```

Los parámetros de esta ecuación relaciona la medida de circunferencia en relación a los días de vida (age).

La función geom_smooth() utiliza la misma función lm() para la determinación de la ecuación de la recta, por ello la línea roja es igual a la obtenida desde los valores de lm().

```{r}
Orange %>%
  ggplot(aes(x = age, y = circumference)) +
  geom_point() + 
  geom_abline(intercept = 17.3997,
              slope = 0.1068,
              col = 'blue') +
  geom_vline(xintercept = 900,
             col = 'green')+
  geom_smooth(method = "lm", col = 'red')
```

¿Cuál es la medida promedio de las naranjas para un árbol de 900 días?

```{r}
dias <- 900
medida <- 0.1068 * dias + 17.3997
medida
```

¿Cuál es la estimación de medidas de frutos para árboles con otras edades?

```{r}
nuevos.dias <- data.frame(age = c(1000,1100, 1200, 1300, 2000))
predict(regresion, nuevos.dias)
```

## Grasas

Datos de registro de nivel de grasa en sangre.

```{r}
grasas <- read.table('http://verso.mat.uam.es/~joser.berrendero/datos/EdadPesoGrasas.txt', header = TRUE)
names(grasas)
```

Cuando existen más de dos variables, es necesario determinar la relación entre ellas, mediante un cuadro de dispersión.

```{r}
# determinar la relación entre las variables
pairs(grasas)
```

```{r}
# determinar el grado de relación lineal
cor(grasas)
```

```{r}
# establecer el modelo de minimos cuadrados
regresionlm <- lm(grasas ~ edad, data = grasas)
summary(regresionlm)
```

### Interpretación

La primera línea es Call: y muestra la formula con la que obtuvimos el modelo.

La segunda es Residuals: y nos da 5 estadísticos sobre la distribución de los residuos del modelo: valores mínimos, 1er, 2do y 3er cuartil y valor máximo.

La media de los residuos siempre es cero, así que es un parámetro de referencia que no aparece en el sumario, pero que nos resulta útil. Si los residuos se distribuyen conforme a una distribución normal deberíamos esperar que la mediana sea 0 o muy cerca de 0, y que el 1er y 3er cuartil sean simétricos. Lo mismo para lo valores mínimos o máximos. Si hay desviaciones notables de media y mediana y no hay simetría entre cuartiles es muy probable que no estemos cumpliendo con algunos de los supuestos de los modelos lineales.

Coefficients: y muestra los coeficientes estimados por el modelo, es decir, los parámetros ocultos β0 y β1 de la ecuación 1. Estas son la ordenada al origen (Intercept) y las pendientes estimadas para cada variable predictora incluída. Para un modelo lineal la relación es directa.

> **Nota**: la ordenada al origen señala el valor de y cuando x=0

En la tercer columna se registra el error estandar para la estimación de cada variable, al que podemos interpretar como el promedio de los residuos. A partir del coeficiente estimado y el error estándar se computa un valor t, un estadístico de la divergencia entre el estimado que produce el modelo y un estimado hipotético con valor 0. Cuanto más alto es el valor t mayor la divergencia entre los coeficientes del modelo y el coeficiente igual a cero. Por último el p-value de la prueba de hipótesis del estadístico t, que indica la probabilidad de obtener un estimado como el que obtuvimos si el coeficiente real fuera 0. Si la probabilidad de este evento es muy baja podemos rechazar la hipótesis de nulidad según la cuál el verdadero estimado es cero.

La última línea muestra el error estándar de los residuos y los valores de R2, múltiple y ajustado. El $R^2$ de 0.712 indica el modelo explica un 71.2% de la varianza de la variable dependiente. Es decir, un 71.2% de la variación en el nivel de grasa se explica por la variación en la edad. El resto de la varianza puede atribuirse al azar o a otras variables que no hemos incorporado al modelo.

(fuente: <https://www.institutomora.edu.mx/testU/SitePages/martinpaladino/modelos_lineales_con_R.html>)

```{r}
regresionlm
```

Quedando la ecuación: $$y = 102.575 + 5.321 * x$$

```{r}
# graficando
# plot(grasas$edad, grasas$grasas, xlab='Edad', ylab='Grasas')
# abline(regresionlm)
grasas %>%
  ggplot(aes(x = edad, y = grasas)) +
  geom_point() +
  geom_abline(slope = coef(regresionlm)[[2]], 
              intercept = coef(regresionlm)[[1]], 
              color = 'red')
```

### Predicciones

Establecer la predicción de contenido de grasa en sangre en base a nuevas edades no contempladas en los datos originales. Creamos un arreglo de nuevos datos y sobre ello se establece los valores predichos según el valor de regresión obtenido.

```{r}
# se genera un arreglo de 11 datos secuenciales de edades de 40 a 50
edades <- data.frame(edad = seq(40,50))
# evaluamos al predicción para esos datos
predict(regresionlm, edades)
```

```{r}
# predicción para un valor especifico
predict(regresionlm, data.frame(edad=90))
```

## Boston house

Registros del valor de viviendas en los suburbios de Boston (USA) en base a diversas variables. Contiene 506 filas y 14 variables.

Fuente: <https://www.kaggle.com/c/boston-housing>

```{r}
# carga de data
data(BostonHousing)
boston <- BostonHousing
str(boston) # mostrar la estructura de los datos.
```

Parametros:

-   crim - nivel de crimen per cápita por ciudad
-   zn - proporción de terrenos residenciales divididos en zonas para lotes de más de 25,000 pies cuadrados
-   indus - proporción de acres de negocios no minoristas por ciudad
-   chas - variable ficticia de Charles River (= 1 si el tramo limita el río, 0 de lo contrario)
-   nox - concentración de óxidos nítricos (partes por 10 millones)
-   rm - número promedio de habitaciones por vivienda
-   age - proporción de unidades ocupadas por sus propietarios construidas antes de 1940
-   dis - Distancias desproporcionadas a cinco centros de empleo de Boston
-   rad - índice de accesibilidad a las autopistas radiales
-   tax - tasa de impuesto a la propiedad de valor completo por USD 10,000
-   ptratio - colegios por localidad
-   b 1000 (B - 0,63)\^ 2, donde B es la proporción de negros por ciudad (¿increíble no?)
-   lstat - porcentaje de estado inferior de la población
-   medv - valor mediano de las viviendas ocupadas por sus propietarios en USD 1000

Este conjunto de datos se utiliza generalmente como ejemplo de Regresión múltiple, si embargo veremos como aplicar este a uno simple. Para lo cual se debe enprimer lugar elegir aquella variable predictora que esté fuertemente correlacionada respecto de las otras.



### Regresión

La librería **corrplot** perite una visualización gráfica de la relación entre variables

```{r}
# descartamos la variable chas por no ser numérica.
corrplot(round(cor(subset(boston, select = -chas)), digits = 3), type = "lower")
````

Como resultado se aprecia a simple vista que **rm** parece más altamente correlacionada que las otras. Lo que se puede verificar numericamente mediante cor().

```{r}
cor(subset(boston, select = -chas))
```

```{r}
bostonregresion <- lm(medv ~ lstat, data = boston)
summary(bostonregresion)
```

```{r}
boston %>%
  ggplot(aes(x = lstat, y = medv)) +
  geom_point() +
  geom_abline(slope = coef(bostonregresion)[[2]], 
              intercept = coef(bostonregresion)[[1]], 
              color = 'red')
```

El modelo no es muy preciso: el valor de la vivienda no se exlica por si solo por la cantidad de habitaciones.

