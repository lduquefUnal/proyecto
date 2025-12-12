# Procesos y pasos clave

Resumen de las tareas realizadas y el flujo seguido para reproducir los experimentos de clasificación híbrida.

## Flujo general
1. **Preparar entorno**  
   - Instalar PyTorch, TorchVision, TorchAudio, NumPy y Matplotlib (ver `modelos/code/requirement.txt`).  
   - Instalar CUDA-Q o usar el simulador `qpp-cpu`.  
   - Configurar Jupyter para ejecutar los notebooks en `proyecto/`.

2. **Preprocesamiento (versión reducida HQNN)**  
   - Transformación MNIST: recorte 28→20 y average pooling 5×5 → imagen 4×4.  
   - Flatten a vector de 16 píxeles para alimentar los 16 qubits.

3. **Circuito cuántico (HQNN reducido)**  
   - Codificación con `Ry(arcsin(x))` en 16 qubits.  
   - PQC escalera con 3 capas de compuertas ZZ (45 bloques CX–RZ–CX).  
   - Medición all-qubit multi-observable (X, Y, Z en cada qubit → 48 features).

4. **Modelo híbrido**  
   - Capa cuántica con parámetros entrenables + classifier lineal `Linear(48, 10)`.  
   - Hiperparámetros por defecto: `n_qubits=16`, `pqc_layers=3`, `batch_size=32`, `epochs=30`, `lr=5e-4` (ajustables en la celda de configuración).

5. **Entrenamiento y validación**  
   - División 80/20 en el dataset reducido de entrenamiento; test en MNIST completo reducido.  
   - Forward/backward de prueba incluido para verificar gradientes de la capa cuántica.  
   - Métricas: pérdida y accuracy en validación por época; gráfica final.

6. **Inferencia y pesos**  
   - Pesos preentrenados disponibles (`hybrid_cnn_mnist_weights*.pth`).  
   - Script de ejemplo: `modelos/code/inference.py` para probar imágenes (e.g., `test_digit.png`).

## Notas y pendientes
- Ajustar `n_samples_prueba` y `epochs` para iteraciones rápidas si el entrenamiento completo es lento.  
- Confirmar que CUDA-Q detecta GPU antes de usar `cudaq.set_target("nvidia")`; en CPU usar `qpp-cpu`.  
- No se incluyen tests automatizados; la validación se realiza ejecutando los notebooks.
