# 1. Usar la imagen base de NVIDIA CUDA Quantum
# Esta imagen ya incluye CUDA, los drivers de NVIDIA y el SDK de CUDA Quantum.
FROM nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 2. Copiar solo el archivo de requerimientos primero
# Esto aprovecha el cache de Docker. Si no cambias los requerimientos,
# Docker no volverá a instalar las librerías en futuras construcciones.
COPY requirements.txt .

# 3. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código de tu aplicación
# En este caso, tu Jupyter Notebook.
COPY red_hibrida.ipynb .

# 5. Exponer el puerto de Jupyter (opcional, para uso interactivo)
# Si quieres ejecutar el notebook dentro del contenedor, esto es útil.
EXPOSE 8888

# 6. Comando por defecto al iniciar el contenedor
# Opción A: Iniciar una terminal interactiva (como tenías).
CMD ["/bin/bash"]

# Opción B: Iniciar Jupyter Lab/Notebook (descomenta la línea que prefieras).
# Esto te permite acceder al notebook desde tu navegador en http://localhost:8888
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]