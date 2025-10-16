# 🔍 Bot de Resoluciones  

Este proyecto implementa un **chatbot para consultar resoluciones en documentos PDF y Word**, utilizando **LangChain**, **FAISS**, **Ollama** y **Streamlit**.  
El sistema permite hacer preguntas en español y recibir respuestas con **citas exactas al archivo y número de resolución correspondiente**.  

---

## 🚀 Requisitos previos  

Antes de ejecutar el chatbot, asegúrate de contar con lo siguiente:  

- **Python 3.10+** instalado  
- **pip** para gestionar dependencias  
- **Ollama** instalado en tu máquina ([descargar desde la web oficial](https://ollama.com))  
- Al menos **40GB de espacio libre en disco** (para descargar y manejar modelos de lenguaje grandes)  
- Memoria RAM recomendada: **16GB+**  

⚠️ **Nota:** El rendimiento dependerá del hardware. En equipos con menos memoria puede que la descarga de modelos falle o el chatbot no funcione correctamente.  

---

## 📦 Instalación  

### 1. Clona este repositorio  

git clone https://github.com/tu-usuario/bot-resoluciones.git
cd bot-resoluciones

### 2. Configura Ollama y el modelo

Antes de ejecutar el chatbot, necesitas descargar el modelo de lenguaje `gemma3:12b` en Ollama:

ollama pull gemma3:12b


# 📝 Nota importante sobre Ollama

Al ejecutar el comando para descargar el modelo `gemma3:12b` en Ollama, puede ocurrir:

- Que Ollama descargue el modelo directamente desde consola.  
- Que Ollama abra la aplicación gráfica. En ese caso:
  1. Selecciona el modelo de lenguaje en la interfaz.  
  2. Haz una consulta sencilla y presiona Enter.  
  3. Espera a que termine la descarga.  
  4. Cierra la aplicación y detén el comando en consola.  

Después de este paso, Ollama ya tendrá el modelo disponible localmente.

---

## ▶️ Ejecución

Si ya se configuro Ollama, simplemente corre el proyecto:

python setup_and_run.py o python3 setup_and_run.py 

Esto descargará automáticamente las dependencias necesarias, iniciará **Streamlit** y podrás acceder al chatbot en tu navegador en la dirección:

http://localhost:8501

