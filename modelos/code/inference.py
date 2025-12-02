import json
import logging
import os
import base64
from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image

# Importamos las clases de los modelos desde nuestro archivo model.py
from model import CNN, Hybrid_QNN

# Configuración del logger
logger = logging.getLogger(__name__)

# --- Funciones requeridas por la plataforma de inferencia (e.g., SageMaker) ---

def model_fn(model_dir):
    """
    Carga TODOS los modelos disponibles desde el directorio de modelos.
    Esta función se ejecuta una vez al iniciar el contenedor.
    """
    logger.info("Buscando y cargando modelos...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    # Mapeo de nombres de archivo a clases de modelo y transformaciones
    # Esto hace que el sistema sea extensible. ¡Solo necesitas añadir una entrada aquí para un nuevo modelo!
    model_map = {
        "hybrid_cnn_v0.0": {"class": Hybrid_QNN, "transform": get_transform_hqnn()},
        "cnn_mnist_weights": {"class": CNN, "transform": get_transform_cnn()}
        # Añade aquí otros modelos: "nombre_modelo": {"class": ClaseModelo, "transform": su_transformacion}
    }

    for model_name, model_info in model_map.items():
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        if os.path.exists(model_path):
            logger.info(f"Cargando modelo '{model_name}' desde {model_path}")
            model_instance = model_info["class"]()
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
            model_instance.to(device).eval()
            models[model_name] = {
                "model": model_instance,
                "transform": model_info["transform"]
            }
        else:
            logger.warning(f"Archivo de pesos no encontrado para el modelo '{model_name}': {model_path}")
            
    if not models:
        raise RuntimeError("¡No se pudo cargar ningún modelo!")
        
    logger.info(f"Modelos cargados exitosamente: {list(models.keys())}")
    return models


def input_fn(request_body, request_content_type):
    """
    Deserializa los datos de entrada. Espera un JSON con "model" y "input".
    El "input" es una imagen en base64.
    """
    logger.info(f"Procesando entrada con content-type: {request_content_type}")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        
        model_name = data.get("model")
        if not model_name:
            raise ValueError("El JSON de entrada debe contener la clave 'model'")
            
        input_b64 = data.get("input")
        if not input_b64:
            raise ValueError("El JSON de entrada debe contener la clave 'input' con la imagen en base64")

        image_data = base64.b64decode(input_b64)
        image = Image.open(BytesIO(image_data))
        
        # Devolvemos el nombre del modelo y la imagen para que predict_fn sepa qué hacer
        return {"model_name": model_name, "image": image}
    else:
        raise ValueError(f"Content-Type no soportado: {request_content_type}")


def predict_fn(input_data, models):
    """
    Realiza la inferencia usando el modelo y la transformación correctos.
    """
    model_name = input_data["model_name"]
    image = input_data["image"]
    
    logger.info(f"Realizando predicción con el modelo: '{model_name}'")
    
    # Verificamos si el modelo solicitado está cargado
    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' no encontrado. Modelos disponibles: {list(models.keys())}")
    
    model_info = models[model_name]
    model = model_info["model"]
    transform = model_info["transform"]
    
    # Aplicamos la transformación específica del modelo
    input_tensor = transform(image).unsqueeze(0)
    
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
    return prediction


def output_fn(prediction, response_content_type):
    """
    Serializa el resultado de la predicción a JSON.
    """
    logger.info(f"Serializando salida para content-type: {response_content_type}")
    if response_content_type == 'application/json':
        # Usamos softmax para asegurar que la salida sean probabilidades
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        
        response = {
            'predicted_class': predicted_idx,
            'probabilities': [f"{p:.6f}" for p in probabilities.tolist()]
        }
        return json.dumps(response)
    else:
        raise ValueError(f"Content-Type no soportado: {response_content_type}")

# --- Funciones de ayuda para las transformaciones ---
# Es bueno separarlas para mantener la claridad, especialmente si cada modelo
# tiene un preprocesamiento diferente.

def get_transform_cnn():
    # Normalización de tu lambda_function.py
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_transform_hqnn():
    # Los modelos cuánticos a menudo no necesitan normalización, o usan una diferente.
    # Ajusta esto según el entrenamiento de tu modelo HQNN.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
        # Podrías necesitar una normalización a [-1, 1] o [0, pi] dependiendo del encoding
        # transforms.Normalize((0.5,), (0.5,)) 
    ])
