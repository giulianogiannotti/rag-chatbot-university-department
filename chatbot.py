import os
import glob
import re
import streamlit as st
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
import numpy as np

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

# --- Cargar vectorstore ---
@st.cache_resource
def load_vectorstore():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DOCS_PATH = os.path.join(BASE_PATH, "RESOLUCIONES_2", "RESOLUCIONES")
    if not os.path.exists(DOCS_PATH):
        st.error(f"No se encontró la carpeta {DOCS_PATH}")
        return None

    doc_paths = glob.glob(f"{DOCS_PATH}/**/*.docx", recursive=True)
    pdf_paths = glob.glob(f"{DOCS_PATH}/**/*.pdf", recursive=True)
    all_paths = doc_paths + pdf_paths

    documents = []
    for path in all_paths:
        documents.extend(load_document(path))

#825 140
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=185)
    splits = splitter.split_documents(documents)


    for chunk in splits:
        res = chunk.metadata.get("resolucion", "Sin identificador")
        name = os.path.basename(chunk.metadata.get("source", "Sin archivo"))
        chunk.page_content = f"[Resolución: {res} | Archivo: {name}]\n{chunk.page_content}"

    embedding_model = HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v2-base-es",
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = FAISS.from_documents(splits, embedding_model)
    return vectorstore, embedding_model

vectorstore, embedding = load_vectorstore()
if vectorstore is None:
    st.stop()

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 15, "fetch_k": 30})

# --- Ollama como LLM ---
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="gemma3:12b",
    temperature=0,
    max_tokens=400
)

# --- Prompt ---
prompt_template = """
Responde obligatoriamente SIEMPRE en español, sin excepciones.
Usa únicamente el contexto proporcionado.
La respuesta debe adaptarse de manera natural a la forma de la pregunta realizada, utilizando un lenguaje similar y coherente (por ejemplo: si preguntan con "¿puedo...?", responder con "Sí, puedes..." o "No, no puedes..."; si preguntan "¿en qué fecha...?", responder directamente con la fecha, etc.).
Redacta la respuesta en un único párrafo fluido, dando primero la información solicitada de forma clara. 
**La respuesta debe mencionar explícitamente el número de resolución correspondiente**. 
Al final, agrega SIEMPRE obligatoriamente la referencia exacta tal como aparece en el contexto, en el mismo formato: [Resolución: ... | Archivo: ...]. 
No reescribas, no resumas ni cambies palabras de la referencia; debe copiarse tal cual.

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
        respuesta_raw = resp["answer"]
        
        if "Respuesta detallada con cita:" in respuesta_raw:
            respuesta_final = respuesta_raw.split("Respuesta detallada con cita:")[-1].strip()
        else:
            respuesta_final = respuesta_raw.strip()
        
        st.session_state.history.append({"question": query, "answer": respuesta_final})



# --- Mostrar historial ---
for qa in st.session_state.history[::-1]:
    st.markdown(f"**Pregunta:** {qa['question']}")
    st.markdown(f"**Respuesta:** {qa['answer']}")
    st.markdown("---")
