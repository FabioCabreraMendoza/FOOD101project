# 🍔 Food-101 Classifier: Deep Learning con MLOps

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)

Este proyecto implementa un sistema de visión por computadora "End-to-End" capaz de clasificar imágenes de alimentos en **101 categorías** distintas. Utiliza técnicas avanzadas de **Transfer Learning** y **Fine-Tuning** sobre la arquitectura MobileNetV2, integrando **MLflow** para el seguimiento de experimentos y **Gradio** para el despliegue de una interfaz de usuario.

---

## 📋 Tabla de Contenidos
- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Arquitectura y Metodología](#-arquitectura-y-metodología)
- [Resultados y Métricas](#-resultados-y-métricas)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Instalación y Uso](#-instalación-y-uso)
- [Estructura del Repositorio](#-estructura-del-repositorio)

---

## 📖 Descripción del Proyecto

El objetivo principal es resolver el problema de la clasificación de alimentos (alta variabilidad intra-clase) utilizando el dataset estándar **Food-101** (5GB de datos). 

El proyecto no solo se enfoca en el modelo, sino en el ciclo de vida de **MLOps**:
1.  **ETL:** Carga eficiente de datos usando `tf.data` y `tensorflow_datasets`.
2.  **Tracking:** Registro automático de parámetros, métricas y artefactos con MLflow.
3.  **Deployment:** Interfaz interactiva para inferencia en tiempo real.

---

## 🧠 Arquitectura y Metodología

Se utilizó una estrategia de **Transfer Learning** en dos fases:

1.  **Fase 1: Feature Extraction (Congelado)**
    * **Base:** MobileNetV2 pre-entrenada en ImageNet (capas congeladas).
    * **Head:** Global Average Pooling + Dropout (0.3) + Dense (Softmax).
    * *Objetivo:* Establecer una línea base estable.

2.  **Fase 2: Fine-Tuning (Descongelado Parcial)**
    * Se descongelaron las **últimas 50 capas** de MobileNetV2.
    * Re-entrenamiento con una tasa de aprendizaje reducida (`1e-5`) para adaptar los filtros a las texturas específicas de los alimentos sin destruir los pesos aprendidos.

---

## 📊 Resultados y Métricas

El modelo fue evaluado utilizando la métrica de **Accuracy** (Exactitud) y **Categorical Crossentropy Loss**.

| Fase del Entrenamiento | Epochs | Accuracy (Validación) | Observaciones |
| :--- | :---: | :---: | :--- |
| **Transfer Learning** | 10 | ~59% | Aprendizaje rápido, sin overfitting. |
| **Fine-Tuning** | +10 | **~71%** | Mejora significativa del 12% al adaptar características. |

> **Nota:** El modelo final alcanza un **70.7% de precisión**, un rendimiento competitivo considerando la eficiencia de MobileNetV2.


---

## 🛠 Tecnologías Utilizadas

* **Google Colab:** Entorno de ejecución (GPU T4).
* **TensorFlow / Keras:** Framework de Deep Learning.
* **MobileNetV2:** Arquitectura de red neuronal ligera.
* **MLflow:** Gestión del ciclo de vida de ML (Tracking).
* **Pyngrok:** Túnel para exponer el servidor de MLflow desde Colab.
* **Gradio:** Creación de la interfaz web (Demo).

---

## 🚀 Instalación y Uso

Este proyecto está diseñado para ejecutarse en **Google Colab** para aprovechar la GPU gratuita.

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/FOOD101project.git](https://github.com/TU_USUARIO/FOOD101project.git)
    ```

2.  **Abrir el Notebook:**
    Sube el archivo `PROYECTO_FOOD101_CLEAN.ipynb` a Google Colab.

3.  **Configurar Ngrok:**
    Necesitarás un token gratuito de [Ngrok](https://dashboard.ngrok.com/get-started/your-authtoken). Reemplaza la variable en el código:
    ```python
    NGROK_AUTH_TOKEN = "TU_TOKEN_AQUI"
    ```

4.  **Ejecutar:**
    Corre todas las celdas. El notebook se encargará de:
    * Instalar dependencias.
    * Descargar el dataset Food-101.
    * Entrenar el modelo.
    * Lanzar la UI de Gradio al final.

---
