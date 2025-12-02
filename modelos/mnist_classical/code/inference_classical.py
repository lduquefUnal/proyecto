import json
import logging
import os
import base64
from io import BytesIO

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Importamos solo la clase del modelo Clásico
from code.modelcnn import CNN

# Configuración del logger
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Carga el modelo CLÁSICO (CNN) desde el directorio.
    """
    logger.info("Iniciando la carga del modelo Clásico (CNN)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    model_path = os.path.join(model_dir, "model.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de modelo no encontrado en: {model_path}")

    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    logger.info("Modelo Clásico (CNN) cargado exitosamente.")
    
    model_info = {
        "model": model,
        "transform": get_transform_cnn()
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
        # Asegura que la imagen sea 28x28 y en escala de grises, como espera el modelo.
        image_processed = image_pil.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
        return image_processed
    else:
        raise ValueError(f"Content-Type no soportado: {request_content_type}")

def predict_fn(image, model_info):
    """
    Realiza la inferencia usando el modelo CNN cargado.
    """
    model = model_info["model"]
    transform = model_info["transform"]
    
    logger.info("Aplicando transformación y realizando predicción (Clásica)...")
    input_tensor = transform(image).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction

def output_fn(prediction, response_content_type):
    """
    Serializa el resultado. Aplica Softmax a los logits del CNN.
    """
    logger.info(f"Serializando salida para content-type: {response_content_type}")
    if response_content_type == 'application/json':
        probabilities = F.softmax(prediction[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()

        response = {
            'predicted_class': predicted_idx,
            'probabilities': [f"{p:.6f}" for p in probabilities.tolist()]
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Content-Type no soportado: {response_content_type}")

def get_transform_cnn():
    """Transformación para el modelo CNN, incluye normalización."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
