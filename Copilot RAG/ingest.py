import os
import ssl

# Solución para el error [SSL: CERTIFICATE_VERIFY_FAILED]
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def ingest_docs():
    # Asegurar que el directorio de datos existe
    if not os.path.exists("./data"):
        os.makedirs("./data")
        print("Directorio './data' creado. Añade tus archivos PDF antes de ejecutar nuevamente.")
        return

    # 1. Cargar documentos
    loader = DirectoryLoader("./data", glob="./*.pdf", loader_cls=PyPDFLoader)
    print("⏳ Iniciando la carga de documentos desde ./data...", flush=True)
    docs = loader.load()
    
    if not docs:
        print("⚠️ No se encontraron archivos PDF válidos en './data'. Verifica que los archivos tengan extensión .pdf", flush=True)
        return
    print(f"📄 Cargados {len(docs)} documentos.", flush=True)

    # 2. Dividir en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        print("⚠️ ERROR: No se generó ningún fragmento de texto. Asegúrate de que los PDFs contengan texto seleccionable y no sean solo imágenes.", flush=True)
        return
    print(f"✂️ Documentos divididos en {len(chunks)} fragmentos.", flush=True)
    
    # 3. Crear Embeddings e índice vectorial
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 4. Persistir
    vectorstore.save_local("./vectorstore")
    print(f"Índice vectorial creado y guardado en './vectorstore' con {len(docs)} documentos procesados.")

if __name__ == "__main__":
    ingest_docs()