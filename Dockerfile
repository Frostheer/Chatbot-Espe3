# Base: Mantenemos la versión 3.11 para máxima estabilidad con librerías ML
FROM python:3.12-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# --- PASO 1: Instalar dependencias del sistema ---
# Esenciales para compilar FAISS, gunicorn y otras librerías complejas.
RUN apt-get update && \
    apt-get install -y build-essential \
                       python3-dev \
                       cmake \
                       libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de dependencias
COPY requirements.txt .

# --- PASO 2: Instalación limpia de Python (incluye tu upgrade de pip) ---
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


COPY .env .env
# Copiamos los datos RAG y el código (incluye 'data/' como lo solicitaste)
COPY data /app/data
COPY . .

# Puerto interno estándar de Gunicorn (mapearemos 5000:8000 en el 'docker run' local)
EXPOSE 8000

# --- PASO 3: Comando de inicio fijo ---
# Ejecuta Gunicorn como un módulo de Python para evitar el error "$PATH"
# Usamos el puerto 8000 y apuntamos a la aplicación 'app' en el archivo 'app_web.py'
CMD ["python", "-m", "gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
