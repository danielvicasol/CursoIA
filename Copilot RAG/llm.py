import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Configura el modelo Llama 3 en Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No se encontró GROQ_API_KEY en las variables de entorno. "
                         "Asegúrate de tener un archivo .env configurado.")

    # Obtenemos el modelo del entorno
    model = os.getenv("GROQ_MODEL_NAME")
    
    # Validamos si el modelo es el antiguo o si no está definido
    # Si es el modelo retirado 'llama3-70b-8192', lo forzamos al nuevo
    if not model or "llama3-70b-8192" in model:
        model = "llama-3.3-70b-versatile"
        print(f"⚠️ Aviso: Se detectó un modelo obsoleto. Cambiando automáticamente a: {model}")

    print(f"--- 🚀 Inicializando Groq con el modelo: {model} ---")

    return ChatGroq(
        model_name=model,
        temperature=0,
        groq_api_key=api_key
    )