# project_transformers_contrastive
#  Proyecto: Detección de autoría literaria mediante aprendizaje contrastivo supervisado y modelos Transformer.

---

##  Resumen general

El proyecto tiene como objetivo desarrollar un modelo capaz de **identificar el autor probable de un fragmento literario** a partir de su **estilo lingüístico, tono emocional y características semánticas**, utilizando técnicas modernas de **aprendizaje profundo y representación contrastiva**.

A diferencia de los enfoques clásicos basados en conteo de palabras o características lingüísticas manuales, este modelo emplea **Transformers fine-tuned** para aprender un **espacio de representaciones estilísticas**, donde fragmentos del mismo autor se agrupan naturalmente por similitud, y fragmentos de autores diferentes se separan.

---

##  Arquitectura general del sistema

1. **Recopilación y preparación de datos**

   * Se seleccionan textos literarios (novelas o relatos) de varios autores.
   * Se dividen en fragmentos de tamaño controlado (≈200–300 palabras).
   * Cada fragmento se etiqueta con su autor.
   * Se reserva un conjunto de validación y prueba con autores conocidos y desconocidos.

2. **Codificación con Transformer y fine-tuning contrastivo**

   * Se emplea un modelo Transformer preentrenado (p. ej. `distilbert-base-multilingual-cased` o `all-MiniLM-L6-v2`).
   * Se le añade una capa de proyección para obtener embeddings de menor dimensión.
   * Se realiza **fine-tuning mediante aprendizaje contrastivo supervisado (Supervised Contrastive Learning)**:

     * Fragmentos del mismo autor se acercan en el espacio de representación.
     * Fragmentos de autores distintos se alejan.
   * Como resultado, el modelo aprende a representar cada fragmento por su **estilo y estructura literaria**, más que por su contenido explícito.

3. **Clasificación y detección open-set**

   * Sobre los embeddings resultantes se entrena un **clasificador probabilístico** (por ejemplo, `LogisticRegression` con calibración).
   * Paralelamente, se calculan **centroides** de cada autor (media de sus embeddings).
   * En inferencia:

     * El modelo devuelve **probabilidades** de pertenencia a cada autor conocido.
     * Calcula también la **similitud coseno** con los centroides.
     * Si la similitud máxima es inferior a un umbral calibrado, el fragmento se marca como **autor desconocido**, devolviendo además los autores más similares estilísticamente.

4. **Interpretabilidad y análisis de estilo**

   * Se aplican métodos de explicabilidad como **Integrated Gradients** o análisis de **mapas de atención**, para identificar qué palabras o frases influyen más en la decisión del modelo.
   * Esto permite analizar **qué rasgos textuales** definen el estilo de cada autor según el modelo (léxico, sintaxis, ritmo, tono emocional, etc.).

---

##  Resultados esperados

* El modelo aprende un **espacio estilístico literario** donde los autores se agrupan por afinidad.
* Permite **clasificar fragmentos** por autor con buenas tasas de acierto, incluso con textos nuevos.
* Detecta cuando un fragmento **no pertenece a ningún autor conocido** y propone autores similares.
* Ofrece interpretaciones visuales y lingüísticas sobre **qué elementos definen el estilo de un autor**.

---

##  Métricas y evaluación

* **Precisión / Recall / F1** sobre autores conocidos.
* **ROC-AUC** para detección de autores desconocidos (open-set).
* **Visualización t-SNE / PCA** de embeddings para mostrar agrupaciones estilísticas.
* **Evaluación cualitativa**: análisis de fragmentos clasificados correctamente / incorrectamente y explicación mediante atribución de gradientes.

---

##  Innovaciones y valor del proyecto

* Integra **aprendizaje contrastivo supervisado** con **modelos Transformer** aplicados a literatura.
* Propone una arquitectura **open-set**, capaz de reconocer estilos no vistos.
* Aporta una **dimensión interpretativa**: el modelo no solo clasifica, sino que explica qué elementos estilísticos sustentan su decisión.
* Combina lingüística, análisis literario y deep learning moderno, todo con **bajo coste computacional** (se puede entrenar con CPU o GPU ligera usando modelos *Distil* o *MiniLM*).

---

##  En una frase final

> El proyecto desarrolla un sistema basado en Transformers y aprendizaje contrastivo supervisado que aprende a representar y reconocer estilos literarios, permitiendo identificar autores, detectar fragmentos de autoría desconocida y analizar los rasgos lingüísticos que definen la firma estilística de cada escritor.

---

✨ En resumen:

> dataset → encoder → embeddings contrastivos → análisis de similitud → explicabilidad textual.
