import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import logging
import os

# Configurar logs para ver el progreso
logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# =========================
# 1️⃣ Cargar dataset
# =========================
df = pd.read_csv("dataset_libros.csv")
df = df.dropna(subset=['autor', 'fragmento'])
autores = df['autor'].unique()
print(f" Autores detectados: {len(autores)}")

# =========================
# 2️⃣ Generar pares (Data Augmentation con Ventana)
# =========================
# Aumentamos el límite ya que ahora generamos más combinaciones
MAX_PARES_POR_AUTOR = 40000 
WINDOW_SIZE = 3  # Emparejar con los 3 siguientes (antes era solo 1)

pos_pairs = []  # Lista de (texto1, texto2, label=1)
neg_pairs = []  # Lista de (texto1, texto2, label=0) para evaluación

print(f" Generando pares con ventana deslizante de {WINDOW_SIZE}...")

for autor in autores:
    textos = df[df['autor'] == autor]['fragmento'].tolist()
    
    # Generar pares positivos (Mismo autor)
    pares_autor = []
    for i in range(len(textos) - WINDOW_SIZE):
        for j in range(1, WINDOW_SIZE + 1):
            # Emparejamos t[i] con t[i+1], t[i+2], etc.
            if i + j < len(textos):
                pares_autor.append((textos[i], textos[i+j]))
    
    # Mezclar y recortar para no explotar la memoria
    random.shuffle(pares_autor)
    pares_autor = pares_autor[:MAX_PARES_POR_AUTOR]
    
    for t1, t2 in pares_autor:
        pos_pairs.append(InputExample(texts=[t1, t2], label=1))

print(f" Total de pares positivos generados: {len(pos_pairs)}")

# =========================
# 3️⃣ Crear Split de Validación (Para detectar Overfitting)
# =========================
print(" Separando set de Entrenamiento (90%) y Validación (10%)...")
train_examples, val_examples_raw = train_test_split(pos_pairs, test_size=0.1, random_state=42)

# Crear DataLoader de entrenamiento
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# =========================
# 4️⃣ Construir Evaluador (La clave para el Early Stopping)
# =========================
# Para evaluar bien, necesitamos también pares NEGATIVOS (diferentes autores) en validación.
# Vamos a generar algunos pares negativos artificiales usando los textos de validación.

print(" Construyendo evaluador con pares positivos y negativos...")
val_pos = []
val_neg = []

# Extraer textos de los ejemplos de validación positivos
textos_val_flat = []
for ex in val_examples_raw:
    val_pos.append(InputExample(texts=ex.texts, label=1))
    textos_val_flat.extend(ex.texts)

# Generar pares negativos aleatorios (textos al azar que seguro son de autores distintos o lejanos)
# Nota: Esto es una aproximación rápida.
random.shuffle(textos_val_flat)
for i in range(0, len(textos_val_flat)-1, 2):
    # Asumimos que al mezclar todo, la probabilidad de que sean del mismo autor es baja
    # Para ser estrictos deberíamos chequear el autor, pero para monitoreo de loss sirve.
    val_neg.append(InputExample(texts=[textos_val_flat[i], textos_val_flat[i+1]], label=0))

# Juntar para el evaluador
eval_examples = val_pos + val_neg[:len(val_pos)] # Balanceado
evaluator = BinaryClassificationEvaluator.from_input_examples(
    eval_examples, 
    batch_size=16,
    name='dev_evaluator',
    show_progress_bar=True
)

# =========================
# 5️⃣ Configurar Modelo
# =========================
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

# Congelar capas (Fine-tuning eficiente)
for name, param in model.named_parameters():
    if "transformer.layer.4" in name or "transformer.layer.5" in name or "pooler" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

train_loss = losses.MultipleNegativesRankingLoss(model)

# =========================
# 6️⃣ Entrenar con Monitoreo
# =========================
output_path = "./modelo_contrastivo_autores_final"

print(" Entrenando con Early Stopping (guardando el mejor modelo)...")
# SBERT no tiene "stop" automático nativo simple, pero save_best_model=True
# asegura que si hay overfitting (la loss sube), tú te quedas con el checkpoint anterior.

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=10,                 # Ponemos más epochs, total pararemos si no mejora
    evaluation_steps=500,      # Evaluar cada 500 pasos
    warmup_steps=100,
    output_path=output_path,
    save_best_model=True,      # <--- ESTO EVITA EL OVERFITTING (Guarda el mejor)
    show_progress_bar=True
)

print(f" Entrenamiento finalizado. El mejor modelo está en: {output_path}")

# =========================
# 7️⃣ Recargar el MEJOR modelo para generar centroides
# =========================
# Es importante recargar, porque el objeto 'model' actual podría ser el de la última epoch (overfitted)
model = SentenceTransformer(output_path)

centroides = []
autores_list = []

print(" Generando centroides con el modelo optimizado...")
for autor in autores:
    textos = df[df['autor'] == autor]['fragmento'].tolist()
    if not textos: continue
    emb = model.encode(textos, batch_size=32, show_progress_bar=False)
    centroide = np.mean(emb, axis=0)
    centroides.append(centroide)
    autores_list.append(autor)

centroides_df = pd.DataFrame(centroides)
centroides_df.insert(0, 'autor', autores_list)
centroides_df.to_csv("centroides_autores_final.csv", index=False)

# =========================
# 8️⃣ Evaluación Final Detallada (Metrics for Paper)
# =========================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print(" Evaluando capacidad de clasificación con el MEJOR modelo guardado...")

# 1. Preparar datos y etiquetas
le = LabelEncoder()
y = le.fit_transform(df["autor"])

# 2. Codificar con barra de progreso
X = model.encode(df["fragmento"].tolist(), show_progress_bar=True)

# 3. Split Stratified (mismas proporciones que el entrenamiento original)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 4. Entrenar el clasificador lineal (Logistic Regression)
# Aumentamos max_iter para evitar warnings de convergencia
clf = LogisticRegression(max_iter=3000, C=1.0) 
clf.fit(X_train, y_train)

# 5. Predecir
y_pred = clf.predict(X_test)

# ---------------------------------------------------------
# CALCULAR MÉTRICAS PARA EL PAPER
# ---------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("="*40)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Ver desglose por autor si tienes curiosidad
# print("\nDesglose por autor:")
# print(classification_report(y_test, y_pred, target_names=le.classes_))