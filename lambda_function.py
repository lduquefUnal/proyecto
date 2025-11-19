import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

# --- 1. Definir la arquitectura del modelo ---
# ¡IMPORTANTE! Esta clase debe ser idéntica a la que usaste para entrenar.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# --- 2. Cargar el modelo fuera del handler ---
# Esto permite que el modelo se cargue solo una vez durante el "arranque en frío" (cold start)
# y se reutilice en invocaciones posteriores, lo cual es mucho más eficiente.
device = "cpu" # Lambda no tiene GPU por defecto
model = NeuralNetwork().to(device)
# El modelo se carga desde el mismo directorio donde está la función Lambda.
model.load_state_dict(torch.load("mnist_model_weights.pth", map_location=device))
model.eval() # Poner el modelo en modo de evaluación

# --- 3. Definir las transformaciones de la imagen ---
# Deben ser las mismas que en el entrenamiento, excepto que la entrada será una imagen PIL.
transform = transforms.Compose([
    transforms.Grayscale(), # Asegurar que la imagen sea de un solo canal
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --- 4. El handler principal de la función Lambda ---
def lambda_handler(event, context):
    try:
        # Extraer la imagen codificada en base64 del cuerpo del evento
        body = json.loads(event.get('body', '{}'))
        if 'image' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No se encontró la clave "image" en el cuerpo de la solicitud.'})
            }
            
        image_data = base64.b64decode(body['image'])
        image = Image.open(io.BytesIO(image_data))

        # Preprocesar la imagen
        image_tensor = transform(image).unsqueeze(0) # Añadir una dimensión de lote (batch)

        # Realizar la inferencia
        with torch.no_grad():
            output = model(image_tensor.to(device))
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_idx = output.argmax(1).item()
            
            # Crear la respuesta
            response = {
                'predicted_digit': predicted_idx,
                'probabilities': {str(i): prob.item() for i, prob in enumerate(probabilities)}
            }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
