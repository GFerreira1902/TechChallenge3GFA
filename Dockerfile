FROM python:3.11-slim

WORKDIR /app

# Copia dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto inteiro
COPY . .

# Expõe a porta do Jupyter
EXPOSE 8888

# Inicia o Jupyter sem autenticação (ambiente local/estudo)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
