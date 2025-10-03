import subprocess
import sys

# Instala los paquetes de Python listados en requirements.txt
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# Ejecuta la app de Streamlit
subprocess.check_call([sys.executable, "-m", "streamlit", "run", "chatbot.py"])
