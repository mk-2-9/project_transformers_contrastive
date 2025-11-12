# Proyecto: Detección de autoría literaria mediante aprendizaje contrastivo y modelos Transformer

---

## Resumen general

Este proyecto desarrolla un modelo de **detección de autoría literaria** capaz de identificar el autor probable de un fragmento textual en función de su **estilo lingüístico**, **tono emocional** y **estructura semántica**, empleando técnicas modernas de **deep learning** y **aprendizaje contrastivo supervisado**.

A diferencia de los enfoques clásicos (n-gramas, conteo de palabras o análisis manual de estilo), este sistema utiliza **modelos Transformer fine-tuned** para aprender un **espacio de representaciones estilísticas**, donde los textos de un mismo autor se agrupan por similitud y los de autores diferentes se separan.

---

## Arquitectura general del sistema

### 1️ Preparación del dataset
- Se recopilan novelas o textos literarios de varios autores (mínimo 2 obras por autor).
- Se dividen en fragmentos de ≈200–300 palabras.
- Cada fragmento se etiqueta con el nombre del autor.
- Se reserva un conjunto de validación y test.

### 2️ Codificación y fine-tuning contrastivo
- Se emplea un modelo preentrenado de Hugging Face (`distiluse-base-multilingual-cased-v2`).
- Se realiza un **fine-tuning parcial** (solo las últimas capas) con **aprendizaje contrastivo supervisado**:
  - Los fragmentos del mismo autor se acercan en el espacio de representación.
  - Los de autores diferentes se alejan.
- Como resultado, se obtiene un modelo que codifica el **estilo literario** en embeddings vectoriales.

### 3️ Clasificación y detección *open-set*
- Sobre los embeddings resultantes se entrena un **clasificador probabilístico** (`LogisticRegression` o `SVM`).
- Se calculan **centroides por autor** para medir similitudes.
- En inferencia:
  - El modelo devuelve **probabilidades** de pertenencia a cada autor conocido.
  - Si la similitud máxima < umbral → el texto se clasifica como **autor desconocido**, mostrando autores estilísticamente cercanos.

### 4️ Interpretabilidad
- Se usan métodos como **Integrated Gradients** y **mapas de atención** para explicar qué partes del texto influyen más en las decisiones del modelo.
- Permite analizar los rasgos textuales que definen el estilo de cada autor (léxico, ritmo, tono emocional, sintaxis...).

---

## Implementación técnica

### Entrenamiento contrastivo

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd, random

# Cargar dataset
df = pd.read_csv("dataset_libros.csv")

# Crear pares positivos (mismo autor)
train_examples = []
for autor in df['autor'].unique():
    textos = df[df['autor']==autor]['fragmento'].tolist()
    if len(textos) < 2: continue
    for _ in range(min(200, len(textos)//2)):
        a, b = random.sample(textos, 2)
        train_examples.append(InputExample(texts=[a, b]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Cargar modelo base multilingüe
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Congelar capas (solo entrenar las últimas)
for name, param in model.named_parameters():
    if not any(layer in name for layer in ["transformer.layer.4", "transformer.layer.5", "dense", "pooling"]):
        param.requires_grad = False

# Pérdida contrastiva
train_loss = losses.MultipleNegativesRankingLoss(model)

# Entrenamiento
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100,
    output_path="./modelo_contrastivo_autores"
)
