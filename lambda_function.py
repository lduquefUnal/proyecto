# lambda_function.py

import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import io
import boto3
import os

# --- 1. Definición del Modelo ---
# ¡IMPORTANTE! Esta clase debe ser una copia exacta de la que usaste para entrenar.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 5 * 5, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits

# --- 2. Carga del Modelo (con optimización para Lambda) ---
# Descargamos el modelo desde S3 solo si no está ya en el entorno de ejecución de Lambda.
# Lambda puede reutilizar entornos, por lo que esto evita descargas repetidas.

S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME') # Nombre del bucket desde variables de entorno
MODEL_KEY = os.environ.get('MODEL_KEY')           # Nombre del archivo en S3 (e.g., 'cnn_mnist_weights.pth')
MODEL_PATH = f"/tmp/{MODEL_KEY}"

# Instanciamos la arquitectura del modelo
model = CNN()

# Solo descargamos si el archivo no existe en el directorio /tmp de Lambda
if not os.path.exists(MODEL_PATH):
    print(f"Descargando modelo desde s3://{S3_BUCKET_NAME}/{MODEL_KEY}")
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET_NAME, MODEL_KEY, MODEL_PATH)

# Cargamos los pesos en la CPU (Lambda no tiene GPU por defecto)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval() # ¡Muy importante! Pone el modelo en modo de evaluación.

# --- 3. Función de preprocesamiento de la imagen ---
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# --- 4. El Handler Principal de Lambda ---
def lambda_handler(event, context):
    try:
        # El frontend enviará la imagen como una cadena base64 en el cuerpo de la solicitud
        image_bytes = base64.b64decode(event['body'])
        
        # Preprocesamos la imagen para que coincida con la entrada del modelo
        tensor = transform_image(image_bytes)
        
        # Realizamos la inferencia
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted_idx = torch.max(probabilities, 0)
            predicted_digit = predicted_idx.item()

        # Devolvemos una respuesta exitosa
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Permite llamadas desde cualquier origen
            },
            'body': json.dumps({
                'predicted_digit': predicted_digit,
                'probabilities': [f"{p:.4f}" for p in probabilities.tolist()]
            })
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

