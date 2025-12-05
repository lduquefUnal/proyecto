import json
import logging
import os
import base64
from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image

# Importamos solo la clase del modelo Híbrido
from code.modelcnn import Hybrid_QNN

# Configuración del logger
logger = logging.getLogger(__name__)

# --- Transformaciones para la imagen de entrada ---
def get_transform_hqnn():
    """
    Define las transformaciones para pre-procesar la imagen de entrada.
    Debe coincidir con las transformaciones usadas durante el entrenamiento.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Valores estándar para MNIST
    ])

def model_fn(model_dir):
    """
    Carga el modelo HÍBRIDO (Hybrid_QNN) desde el directorio.
    """
    logger.info("Iniciando la carga del modelo Híbrido (Hybrid_QNN)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    # --- IMPORTANTE ---
    # El script espera 'model.pth'. Asegúrate de que tu archivo de pesos
    # 'hybrid_cnn_mnist_weights_cpu_v2.0.pth' sea renombrado a 'model.pth'
    # antes de crear el archivo tar.gz.
    model_path = os.path.join(model_dir, "model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de modelo no encontrado en: {model_path}")

    model = Hybrid_QNN()
    # Usamos weights_only=True para mayor seguridad, como recomienda la advertencia de PyTorch.
    # Esto asume que el archivo .pth solo contiene los pesos y no código arbitrario.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    logger.info("Modelo Híbrido (Hybrid_QNN) cargado exitosamente.")
    
    model_info = {
        "model": model,
        "transform": get_transform_hqnn()
    }
    return model_info

def input_fn(request_body, request_content_type):
    """
    Deserializa los datos de entrada. Espera un JSON con "input" en base64.
    """
    logger.info(f"Procesando entrada con content-type: {request_content_type}")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        input_b64 = data.get("input")
        if not input_b64:
            raise ValueError("El JSON de entrada debe contener la clave 'input' con la imagen en base64")

        image_data = base64.b64decode(input_b64)
        image_pil = Image.open(BytesIO(image_data))
        
        # --- Pre-procesamiento robusto ---
        # 1. Convertir a escala de grises ('L' mode en PIL)
        # 2. Redimensionar a 28x28, como espera el modelo
        image_processed = image_pil.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        
        return image_processed
    else:
        raise ValueError(f"Content-Type no soportado: {request_content_type}")

def predict_fn(image, model_info):
    """
    Realiza la inferencia usando el modelo Hybrid_QNN cargado.
    """
    model = model_info["model"]
    transform = model_info["transform"]
    
    logger.info("Aplicando transformación y realizando predicción (Híbrida)...")
    input_tensor = transform(image).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction

def output_fn(prediction, response_content_type):
    """
    Serializa el resultado. La salida del modelo ya son probabilidades.
    """
    logger.info(f"Serializando salida para content-type: {response_content_type}")
    if response_content_type == 'application/json':
        probabilities = prediction[0]
        predicted_idx = torch.argmax(probabilities).item()

        response = {
            'predicted_class': predicted_idx,
            'probabilities': [f"{p:.6f}" for p in probabilities.tolist()]
        }
        
        # Devuelve la respuesta como una cadena JSON, como esperaría un endpoint real.
        return json.dumps(response)