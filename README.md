# Proyecto: Clasificación híbrida cuántico–clásica en MNIST

Guía breve para reproducir los experimentos del repositorio (modelos clásico, híbrido y versión reducida inspirada en el paper de HQNN).

## Contenido principal
- `red_clasica.ipynb`: pipeline CNN clásica con MNIST.
- `red_hibrida.ipynb`: modelo híbrido inicial (CNN + capa cuántica).
- `red_hibrida_reducida.ipynb`: HQNN reducido (imágenes 4×4, 16 qubits, 45 compuertas ZZ, medición X/Y/Z).
- `modelos/code/inference.py`: ejemplo de inferencia con pesos exportados.
- Pesos preentrenados: `hybrid_cnn_mnist_weights*.pth`.
- Paper de referencia: `pdf/Multi-Classification Hybrid.pdf`.

## Prerrequisitos
- Python 3.10+ recomendado.
- Jupyter Notebook o JupyterLab.
- PyTorch, TorchVision, TorchAudio, NumPy, Matplotlib (ver `modelos/code/requirement.txt` para versiones fijadas).
- Para los cuadernos cuánticos: CUDA-Q (o simulador `qpp-cpu`). Instalar según la guía oficial de NVIDIA CUDA-Q antes de ejecutar las celdas cuánticas.

## Instalación de dependencias (entorno local)
```bash
cd proyecto
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\\Scripts\\activate
pip install -r modelos/code/requirement.txt
# Para CUDA-Q, sigue las instrucciones oficiales (no incluida en requirements).
```

## Reproducir experimentos
1) Abre Jupyter en la carpeta `proyecto`:
```bash
jupyter notebook
```
2) Ejecuta el cuaderno deseado:
   - `red_clasica.ipynb`: entrenamiento CNN 28×28 (referencia clásica).
   - `red_hibrida.ipynb`: versión híbrida inicial.
   - `red_hibrida_reducida.ipynb`: HQNN reducido del paper. Usa:
     - Reducción de imagen: recorte 28→20 y average pooling 5×5 → 4×4.
     - 16 qubits, 3 capas en escalera (45 compuertas ZZ descompuestas CX–RZ–CX).
     - Medición all-qubit X/Y/Z (48 features) y clasificador lineal.
     - Hiperparámetros en la celda de configuración (ajusta `n_samples_prueba`, `epochs`, `batch_size` si quieres ejecuciones rápidas).
3) Los datasets MNIST se descargan automáticamente vía TorchVision.

## Inferencia con pesos guardados
Ejemplo usando `inference.py` (ajusta la ruta del peso que quieras probar):
```bash
cd proyecto/modelos/code
python inference.py --weights ../../hybrid_cnn_mnist_weights_cpu.pth --image ../../test_digit.png
```

## Notas
- No se ejecutan pruebas automáticas; valida manualmente en Jupyter.
- Si usas GPU, asegúrate de que PyTorch detecta CUDA y que CUDA-Q está configurado con `cudaq.set_target("nvidia")`. Para CPU, usa `cudaq.set_target("qpp-cpu")`.
