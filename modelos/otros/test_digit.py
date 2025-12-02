import torch
import torchvision
from PIL import Image

print("Descargando el dataset de prueba de MNIST...")
# Descargar el dataset de prueba de MNIST
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True
)

print("Dataset descargado.")

# Tomemos una imagen de ejemplo (la tercera imagen, que es un '1')
image_tensor, label = test_dataset[2]

print(f"Imagen seleccionada. Es un dígito: {label}")

# El dataset devuelve un objeto PIL.Image directamente.
# Guardamos la imagen en el disco.
file_name = 'test_digit.png'
image_tensor.save(file_name)

print(f"¡Listo! La imagen ha sido guardada como '{file_name}' en el directorio actual.")
