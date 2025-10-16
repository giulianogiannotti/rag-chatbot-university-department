import os
import glob
import re
import pickle
import streamlit as st
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document as LCDocument
from docx import Document
import pdfplumber
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuración de página ---
st.set_page_config(page_title="Bot de Resoluciones", layout="wide")
st.title("🔍 Bot de Resoluciones")
st.write("Consultá resoluciones y obtené respuestas detalladas con cita de archivos.")

# --- Patrones de resoluciones ---
PATRONES = [
    re.compile(r'(?P<letras>[A-Z]{1,5})-(?P<num1>\d{3})[-\.](?P<num2>\d{2,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})[-\.](?P<letras2>[A-Z]{2,4})[-\.](?P<num>\d{1,4})', re.IGNORECASE),
    re.compile(r'(?P<letras1>[A-Z]{1})-(?P<letras2>[A-Z]{3})[-\.](?P<num>\d{2,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})[-\.](?P<num1>\d{1,4})[-\.](?P<num2>\d{1,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})[-\.](?P<num1>\d{1,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})\s+(?P<num1>\d{3})[-\.](?P<num2>\d{2,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})-?\s*(?P<num1>\d{3})[-\.]{1,2}(?P<num2>\d{2,4})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{1,2})-(?P<num1>\d{3})[-\.](?P<num2>\d{2})', re.IGNORECASE),
    re.compile(r'(?P<letras>[A-Z]{3,5})-\.(?P<num1>\d{2,4})', re.IGNORECASE),
]

def extraer_metadato_resolucion(nombre_archivo):
    for patron in PATRONES:
        match = patron.search(nombre_archivo)
        if match:
            return match.group(0)
    return nombre_archivo

# --- Procesamiento de tablas ---
def procesar_tablas_docx(path):
    frases = []
    try:
        doc = Document(path)
        for table in doc.tables:
            for row in table.rows:
                celdas = [cell.text.strip() for cell in row.cells]
                if any(celdas):
                    frases.append(" | ".join(celdas))
    except Exception as e:
        print(f"⚠️ Error procesando tablas en {path}: {e}")
    return frases

def procesar_tablas_pdf(path):
    frases = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                tablas = page.extract_tables()
                for tabla in tablas:
                    for fila in tabla:
                        if any(fila):
                            fila_limpia = [str(x).strip() for x in fila if x]
                            frases.append(" | ".join(fila_limpia))
    except Exception as e:
        print(f"⚠️ Error procesando tablas en {path}: {e}")
    return frases

# --- Cargar documentos ---
def load_document(file_path):
    try:
        extra_docs = []
        if file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
            extra_docs = loader.load()
            frases_tablas = procesar_tablas_docx(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            extra_docs = loader.load()
            frases_tablas = procesar_tablas_pdf(file_path)
        else:
            return []

        nombre_archivo = os.path.basename(file_path)
        identificador = extraer_metadato_resolucion(nombre_archivo)

        for doc in extra_docs:
            doc.metadata["resolucion"] = identificador
            doc.metadata["source"] = nombre_archivo

        tabla_docs = [
            LCDocument(
                page_content=f"[Tabla | Resolución: {identificador} | Archivo: {nombre_archivo}] {frase}",
                metadata={"resolucion": identificador, "source": nombre_archivo}
            ) for frase in frases_tablas
        ]
        return extra_docs + tabla_docs
    except Exception as e:
        print(f"Error al cargar {file_path}: {e}")
        return []

# --- Guardar y cargar índice ---
def guardar_estado_indices(estado_path, estado):
    with open(estado_path, "wb") as f:
        pickle.dump(estado, f)

def cargar_estado_indices(estado_path):
    if os.path.exists(estado_path):
        with open(estado_path, "rb") as f:
            return pickle.load(f)
    return {}

# --- Cargar vectorstore ---
@st.cache_resource
def load_vectorstore():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DOCS_PATH = os.path.join(BASE_PATH, "RESOLUCIONES_2", "RESOLUCIONES")
    INDEX_PATH = os.path.join(BASE_PATH, "faiss_index")
    ESTADO_PATH = os.path.join(BASE_PATH, "estado_indices.pkl")

    os.makedirs(DOCS_PATH, exist_ok=True)
    os.makedirs(INDEX_PATH, exist_ok=True)

    # --- Cargar estado previo ---
    estado_prev = cargar_estado_indices(ESTADO_PATH)

    # --- Buscar documentos nuevos o modificados ---
    all_paths = glob.glob(f"{DOCS_PATH}/**/*.docx", recursive=True) + \
                glob.glob(f"{DOCS_PATH}/**/*.pdf", recursive=True)

    nuevos_docs = []
    estado_actual = {}

    for path in all_paths:
        mod_time = os.path.getmtime(path)
        estado_actual[path] = mod_time
        if path not in estado_prev or estado_prev[path] != mod_time:
            nuevos_docs.append(path)

    # --- Cargar embeddings ---
    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-es",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    # --- Cargar o crear índice ---
    if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Índice FAISS cargado desde disco.")
    else:
        vectorstore = None
        print("⚠️ No se encontró índice FAISS. Se creará uno nuevo.")

    # --- Reindexar solo nuevos documentos ---
    if nuevos_docs:
        print(f"🔄 Reindexando {len(nuevos_docs)} documentos nuevos o modificados...")
        documents = []
        for path in nuevos_docs:
            documents.extend(load_document(path))

        splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=185)
        splits = splitter.split_documents(documents)

        for chunk in splits:
            res = chunk.metadata.get("resolucion", "Sin identificador")
            name = os.path.basename(chunk.metadata.get("source", "Sin archivo"))
            chunk.page_content = f"[Resolución: {res} | Archivo: {name}]\n{chunk.page_content}"

        if vectorstore is None:
            vectorstore = FAISS.from_documents(splits, embedding_model)
        else:
            vectorstore.add_documents(splits)

        vectorstore.save_local(INDEX_PATH)
        guardar_estado_indices(ESTADO_PATH, estado_actual)
        print("💾 Índice FAISS actualizado y guardado.")
    else:
        print("✅ No hay documentos nuevos o modificados.")

    return vectorstore, embedding_model

# --- Cargar índice y configurar retriever ---
vectorstore, embedding = load_vectorstore()
if vectorstore is None:
    st.error("No se pudo cargar el índice FAISS.")
    st.stop()

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 30})

# --- Ollama como LLM ---
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:12b", temperature=0, max_tokens=400)

# --- Prompt ---
prompt_template = """
Responde obligatoriamente SIEMPRE en español, sin excepciones.
Usa únicamente el contexto proporcionado.
La respuesta debe adaptarse de manera natural a la forma de la pregunta realizada.
Menciona explícitamente el número de resolución.
Al final agrega la referencia exacta en el formato: [Resolución: ... | Archivo: ...].
Si no hay información suficiente, responde:
"No se encontró información suficiente en las resoluciones disponibles."

Contexto: {context}
Pregunta: {question}
Respuesta:
"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=False
)

# --- Historial ---
if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Escribí tu pregunta:")

if query:
    with st.spinner("Buscando respuesta..."):
        chat_history = [(qa["question"], qa["answer"]) for qa in st.session_state.history]
        resp = qa_chain.invoke({"question": query, "chat_history": chat_history})
        respuesta_final = resp["answer"].strip()
        st.session_state.history.append({"question": query, "answer": respuesta_final})

# --- Mostrar historial ---
for qa in st.session_state.history[::-1]:
    st.markdown(f"**Pregunta:** {qa['question']}")
    st.markdown(f"**Respuesta:** {qa['answer']}")
    st.markdown("---")
