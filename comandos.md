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


ID DOCKER   :
a11c4e134cfc