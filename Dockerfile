# 1. Usar la imagen base de NVIDIA CUDA Quantum
FROM nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest

# Establecer un directorio de trabajo
WORKDIR /app

# 2. Copiar tu código a la imagen.
COPY app.py /var/task/

# 3. (Opcional) Definir un comando por defecto. 
# Por ejemplo, para ejecutar un script de Python al iniciar el contenedor.
# CMD ["python3", "/app/app.py"]

# Si quieres que inicie una terminal interactiva por defecto, puedes usar:
CMD ["/bin/bash"]