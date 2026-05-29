import streamlit as st
import os
import ssl

# Solución para el error [SSL: CERTIFICATE_VERIFY_FAILED]
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

from llm import get_llm
from rag import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Groq RAG Agent", layout="wide")
st.title("🤖 Agente RAG (Groq + FAISS)")

# --- Inicialización de componentes ---

# Solo cacheamos el modelo de Embeddings (pesado en memoria)
@st.cache_resource(show_spinner="Cargando modelo de lenguaje local...")
def get_embeddings_transformer():
    try:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error al cargar Embeddings: {e}")
        st.stop()

# Solo cacheamos el Vectorstore (I/O de disco)
@st.cache_resource(show_spinner="Cargando base de datos vectorial...")
def get_vectorstore_instance(_embeddings):
    vectorstore_path = "./vectorstore"
    if not os.path.exists(vectorstore_path):
        st.warning(f"⚠️ No se detectó un índice en '{vectorstore_path}'. Asegúrate de tener archivos en /data y ejecutar 'python ingest.py'")
        st.stop()
    try:
        return FAISS.load_local(vectorstore_path, _embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error al cargar el índice vectorial: {e}. Intenta ejecutar 'python ingest.py' nuevamente.")
        st.stop()

# Componentes principales
embeddings = get_embeddings_transformer()
vectorstore = get_vectorstore_instance(embeddings)

# Importante: El LLM y la cadena se crean sin caché para evitar "Closed Client"
llm = get_llm()
chain = get_rag_chain(llm, vectorstore)

# Inicializar memoria de chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inicializar mensajes para Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Interfaz de Usuario ---

# Botón para limpiar el historial de chat
if st.sidebar.button("Limpiar Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun() # Recargar la página para limpiar el chat

# Botón para forzar la recarga del índice vectorial
if st.sidebar.button("🔄 Recargar Documentos"):
    st.cache_resource.clear()
    st.success("Caché limpiado. El índice se recargará.")
    st.rerun()

# Mostrar mensajes previos en la UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Qué deseas consultar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando documentos..."):
            response = chain.invoke({
                "question": prompt, 
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["answer"]
            st.markdown(answer)
            
            if response.get("source_documents"):
                with st.expander("Fuentes"):
                    for doc in response["source_documents"]:
                        st.write(f"- {doc.metadata['source']}")
            
            # Guardar la interacción en el historial (formato LangChain Core)
            st.session_state.chat_history.extend([
                HumanMessage(content=prompt),
                AIMessage(content=answer)
            ])
            
            st.session_state.messages.append({"role": "assistant", "content": answer})