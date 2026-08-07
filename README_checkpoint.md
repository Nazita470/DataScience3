# Checkpoint: Clasificador Supervisado

## Dataset

Se utilizó el mismo corpus de AG News del Módulo 2:

- `ag_news_train.csv`: 8000 noticias.
- `ag_news_test.csv`: 2000 noticias.
- Variables: `text` y `label`.
- Categorías: `Business`, `Sci_Tech`, `Sports`, `World`.

## Pipeline

El pipeline sigue este orden:

1. Carga del train y test.
2. Limpieza del texto.
3. `TfidfVectorizer`.
4. Entrenamiento del clasificador.
5. Predicción sobre test.
6. Evaluación mediante `classification_report`.
7. Matriz de confusión.

### Prevención de Data Leakage

El vectorizador se ajusta exclusivamente con el conjunto de entrenamiento:

```python
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
```

Por lo tanto, el vocabulario y los valores IDF se obtienen solamente a partir de `train`. El conjunto de test nunca participa en el ajuste del vectorizador.

## Experimentos

Se probaron distintas cantidades máximas de características y distintos rangos de n-gramas:

| max_features | ngram_range | Accuracy |
|---:|:---:|---:|
| 5000 | (1, 1) | 0.8870 |
| 10000 | (1, 1) | 0.8895 |
| 10000 | (1, 2) | 0.8895 |
| 20000 | (1, 2) | 0.8910 |

La configuración seleccionada fue:

- `max_features=20000`
- `ngram_range=(1, 2)`

El resultado final fue una accuracy de aproximadamente **89.1%**.

## ¿Por qué Regresión Logística?

Se eligió Regresión Logística porque funciona muy bien como baseline para clasificación de texto representado mediante TF-IDF. Es un modelo lineal, eficiente con matrices dispersas y permite establecer una referencia sólida sin introducir demasiada complejidad.

Para este problema resulta una opción adecuada frente a modelos más complejos porque el espacio de características generado por TF-IDF puede ser muy grande y disperso.

## Análisis preliminar de resultados

El modelo obtuvo aproximadamente:

- `Sports`: F1 = 0.94
- `World`: F1 = 0.90
- `Sci_Tech`: F1 = 0.87
- `Business`: F1 = 0.85

La categoría más difícil fue **Business**, seguida de **Sci_Tech**. Esto se observa en la matriz de confusión porque varias noticias de Business son clasificadas como Sci_Tech o World, mientras que Sports presenta una separación mucho más clara.

Esto tiene sentido porque algunas noticias de negocios pueden contener vocabulario tecnológico o internacional, mientras que las noticias deportivas suelen utilizar términos mucho más específicos del dominio.

## Requisitos

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las librerías utilizadas.
