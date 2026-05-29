# Agente RAG con Streamlit + Groq

## Descripción

Este proyecto implementa un **agente RAG (Retrieval-Augmented Generation)** que permite responder preguntas basadas en documentos locales cargados en un directorio.

El sistema utiliza:
- **Groq API** para inferencia LLM ultra rápida
- **Embeddings** para indexación semántica
- **Vector Store** para recuperación eficiente
- **Streamlit** como interfaz de usuario

---

## Arquitectura del Proyecto

```
c:\Copilot RAG\
│
├── data/                # Documentos fuente
├── vectorstore/         # Índice persistente
│
├── ingest.py            # Script para procesar e indexar documentos
├── rag.py               # Módulo con la lógica de la cadena RAG
├── llm.py               # Módulo para inicializar el LLM de Groq
├── ui.py                # Interfaz de usuario de Streamlit
│
├── requirements.txt
└── README.md
```

---

## Stack Tecnológico

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA3 / Mixtral)
- **Embeddings**: SentenceTransformers
- **Vector Store**: FAISS o Chroma
- **Pipeline**: LangChain (opcional)

---

## Flujo del Sistema

```
Usuario → Pregunta
        → Embedding query
        → Búsqueda en vector DB
        → Recuperación (Top-K chunks)
        → Construcción prompt
        → LLM (Groq)
        → Respuesta contextualizada
```

---

## Instalación

```bash
git clone <repo>
cd "Copilot RAG"

python -m venv venv
.\venv\Scripts\activate  # En Windows (PowerShell)
pip install -r requirements.txt
```

---

## Variables de Entorno

Crea un archivo llamado `.env` en la raíz del proyecto con tu clave:
`GROQ_API_KEY=tu_clave_de_groq_aqui`

---

## Ingesta de Documentos

Coloca los archivos en `/data` y ejecuta:

```bash
python ingest.py
```

---

## Ejecución

```bash
streamlit run ui.py
# O si el comando no es reconocido:
python -m streamlit run ui.py
```

---

## Resumen

Solución RAG rápida, modular y lista para escalar con Groq + Streamlit.
