## crear un entorno virtual
python -m venv qc
source qc/bin/activate

pip install --upgrade pip
pip install jupyter numpy matplotlib
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install cudaq

pip install ipykernel
python -m ipykernel install --user --name=quantum_env

docker pull nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest-base
correr 
docker run -it --name cuda-quantum nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
ID DOCKER   : a11c4e134cfcecaa8ab63e3425a5ece6abc25095684ca9d94fe069e8b0ce13ec

## Flujo de trabajo para habilitar GPU en un contenedor existente

# 1. "Guardar" el estado del contenedor actual (con librerías instaladas) en una nueva imagen.
# Usar el ID del contenedor original.
docker commit a11c4e134cfc cuda-quantum:gpu-ready

# 2. (Opcional pero recomendado) Detener y eliminar el contenedor antiguo para liberar espacio y nombre.
docker stop a11c4e134cfc
docker rm a11c4e134cfc

# 3. Asegurarse de tener el toolkit de NVIDIA para Docker
sudo dnf install -y nvidia-docker2
sudo systemctl restart docker

# 4. Correr un NUEVO contenedor desde la imagen personalizada, ahora con acceso a la GPU.
# Ejecutar desde la carpeta del proyecto.
# El flag --gpus all es para habilitar la GPU.
# El flag -v "$(pwd)":/app monta el directorio actual dentro de /app en el contenedor.
docker run -it --gpus all --name cuda-quantum-gpu -v "$(pwd)":/app cuda-quantum:gpu-ready
a11c4e134cfc