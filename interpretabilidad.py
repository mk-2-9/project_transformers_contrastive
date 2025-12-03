"""
🧠 Módulo de Interpretabilidad de Estilo Literario
--------------------------------------------------
Este módulo permite "abrir la caja negra" del modelo SentenceTransformer
fine-tuneado. Utiliza técnicas de explicabilidad para entender qué rasgos
estilísticos (palabras, puntuación) asocia el modelo a cada autor.

Funcionalidades:
 1. Explicación de Estilo (Integrated Gradients vs Centroide del Autor).
 2. Visualización de Mapas de Atención (Transformer Heads).
 3. Proyección del Espacio Latente (t-SNE de Autores).
 4. Comparación Matemática de Estilos.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import HTML, display

from sentence_transformers import SentenceTransformer
from captum.attr import IntegratedGradients

# Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================
# 1️⃣ Carga de Recursos
# ==============================================================

def cargar_recursos(ruta_modelo="./modelo_contrastivo_autores_1/", ruta_centroides="centroides_autores_1.csv"):
    """
    Carga el modelo y los centroides pre-calculados.
    Retorna: model, tokenizer, df_centroides
    """
    print(f"⏳ Cargando modelo desde: {ruta_modelo}...")
    model = SentenceTransformer(ruta_modelo).to(device)
    model.eval()
    
    # Accedemos al tokenizador y al modelo base (Transformer) para Captum
    tokenizer = model.tokenizer
    
    print(f"⏳ Cargando centroides desde: {ruta_centroides}...")
    try:
        df_centroides = pd.read_csv(ruta_centroides)
        print(f"✅ Recursos cargados. Dispositivo: {device}")
        return model, tokenizer, df_centroides
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo de centroides. Ejecuta primero el train.")
        return model, tokenizer, None

# ==============================================================
# 2️⃣ Explicabilidad del Estilo (Integrated Gradients)
# ==============================================================

def explicar_estilo_autor(texto, autor_objetivo, model, tokenizer, df_centroides):
    """
    Usa Integrated Gradients considerando la proyección de dimensionalidad
    específica de distiluse-base-multilingual (768 -> 512).
    """
    
    # 1. Obtener el vector centroide del autor (512 dimensiones)
    if autor_objetivo not in df_centroides['autor'].values:
        raise ValueError(f"El autor '{autor_objetivo}' no está en los centroides.")
    
    datos_autor = df_centroides[df_centroides['autor'] == autor_objetivo].iloc[0, 1:].values
    centroide = np.array(datos_autor, dtype=np.float32)
    centroide_tensor = torch.tensor(centroide, device=device).unsqueeze(0) # (1, 512)

    # 2. Identificar sub-módulos del modelo
    # model[0] = Transformer (auto_model) -> Salida 768
    # model[1] = Pooling
    # model[2] = Dense (Linear 768->512) + Tanh
    
    base_model = model[0].auto_model
    base_model.to(device)
    embedding_layer = base_model.embeddings.word_embeddings
    
    # Capa de proyección (Dense) específica de este modelo
    # Si el modelo no tiene módulo 2, asumimos que no hay proyección
    projection_layer = model[2].linear if len(model) > 2 else None
    
    encoded = tokenizer(texto, return_tensors='pt', truncation=True, padding=True).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    input_embeddings = embedding_layer(input_ids)
    
    # 3. Función forward que replica EXACTAMENTE la arquitectura del modelo
    def forward_similarity(inputs_embeds, mask):
        # A. Transformer
        outputs = base_model(inputs_embeds=inputs_embeds, attention_mask=mask)
        token_embeddings = outputs.last_hidden_state # (Batch, Seq, 768)
        
        # B. Mean Pooling (Manual)
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_embedding = sum_embeddings / sum_mask # (Batch, 768)
        
        # C. Proyección Densa (768 -> 512)
        # ESTA ES LA PARTE QUE FALTABA Y CAUSABA EL ERROR
        if projection_layer is not None:
            final_embedding = projection_layer(pooled_embedding)
            # distiluse usa Tanh como activación después de la densa
            final_embedding = torch.tanh(final_embedding) 
        else:
            final_embedding = pooled_embedding

        # D. Similitud Coseno con el Centroide (1, 512)
        cos_sim = torch.nn.functional.cosine_similarity(final_embedding, centroide_tensor)
        return cos_sim

    # 4. Calcular atribuciones
    ig = IntegratedGradients(forward_similarity)
    
    attributions = ig.attribute(
        inputs=input_embeddings,
        additional_forward_args=(attention_mask,),
        n_steps=30
    )
    
    importancias = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    
    # Limpieza y Normalización
    tokens_limpios, scores_limpios = [], []
    for t, s in zip(tokens, importancias):
        if t in ["[CLS]", "[SEP]", "[PAD]"]: continue
        tokens_limpios.append(t.replace("##", ""))
        scores_limpios.append(s)

    scores_arr = np.array(scores_limpios)
    max_val = np.max(np.abs(scores_arr))
    norm_scores = scores_arr / (max_val + 1e-9) if max_val > 0 else scores_arr
    
    return tokens_limpios, norm_scores

# ==============================================================
# 3️⃣ Visualización de Texto (Heatmap HTML)
# ==============================================================

def visualizar_texto_coloreado(tokens, scores, autor_objetivo):
    """
    Renderiza HTML con los tokens coloreados.
    Rojo = Rasgo muy propio del autor.
    Azul = Rasgo que aleja del estilo del autor.
    """
    cmap = cm.get_cmap("bwr") # Blue-White-Red colormap
    html_text = f"<h3>Análisis de estilo hacia: {autor_objetivo}</h3>"
    html_text += "<div style='line-height: 1.8; font-family: monospace; font-size: 14px; border:1px solid #ddd; padding:15px;'>"
    
    for token, score in zip(tokens, scores):
        # Mapear score (-1 a 1) a color (0 a 1)
        # Score 0 (neutro) -> 0.5 (blanco)
        color_val = (score + 1) / 2
        rgba = cmap(color_val)
        hex_color = mcolors.rgb2hex(rgba)
        
        # Color de texto (blanco si el fondo es muy oscuro)
        text_color = "white" if abs(score) > 0.5 else "black"
        
        html_text += f"<span style='background-color: {hex_color}; color: {text_color}; padding: 2px 4px; border-radius: 4px; margin-right: 2px;' title='Score: {score:.3f}'>{token}</span>"
        
    html_text += "</div>"
    html_text += "<p style='font-size:12px; color:gray;'>🔴 Rojo: Típico del autor | 🔵 Azul: Atípico / Otro estilo</p>"
    
    display(HTML(html_text))

# ==============================================================
# 4️⃣ Visualización Global (Espacio Latente)
# ==============================================================

def plot_espacio_autores(df_centroides):
    """
    Visualiza dónde se ubican los autores en el espacio vectorial usando t-SNE.
    """
    autores = df_centroides["autor"].values
    vectores = df_centroides.iloc[:, 1:].values
    
    print("🔄 Calculando proyección t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(5, len(autores)-1), random_state=42, init='pca', learning_rate='auto')
    proyeccion = tsne.fit_transform(vectores)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=proyeccion[:,0], y=proyeccion[:,1], hue=autores, s=200, palette="viridis", edgecolor='black')
    
    for i, autor in enumerate(autores):
        plt.text(proyeccion[i,0]+0.2, proyeccion[i,1]+0.2, autor, fontsize=9)
        
    plt.title("Mapa Estilístico de Autores (t-SNE)")
    plt.xlabel("Dimensión 1")
    plt.ylabel("Dimensión 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==============================================================
# 5️⃣ Ejemplo de Uso (Main)
# ==============================================================

if __name__ == "__main__":
    # 1. Cargar
    model, tokenizer, df = cargar_recursos()
    
    if df is not None:
        # 2. Ver mapa global
        plot_espacio_autores(df)
        
        # 3. Elegir un texto de prueba y un autor objetivo
        texto_prueba = "Era una noche oscura y tormentosa, los árboles crujían bajo el viento implacable."
        
        # Selecciona un autor que exista en tu CSV
        autor_test = df['autor'].iloc[0] 
        print(f"\n🔬 Analizando texto vs estilo de: {autor_test}")

        try:
            tokens, scores = explicar_estilo_autor(texto_prueba, autor_test, model, tokenizer, df)
            visualizar_texto_coloreado(tokens, scores, autor_test)
            
            # Gráfico de barras de importancia
            plt.figure(figsize=(12, 3))
            sns.barplot(x=tokens[:20], y=scores[:20], palette="bwr")
            plt.xticks(rotation=45)
            plt.title("Influencia de las primeras palabras en la detección del autor")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error en el análisis: {e}")