FROM python:3.9-slim

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice sorgente
COPY src/ src/
COPY tests/ tests/

# Imposta variabili d'ambiente
ENV PYTHONPATH=/app

# Comando predefinito
CMD ["python", "-m", "src.model_training"]