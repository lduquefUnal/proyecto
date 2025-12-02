import torch
import torch.nn as nn

# --- Modelo CNN Clásico (tomado de tu lambda_function.py) ---
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

# --- Modelo Híbrido Cuántico-Clásico (HQNN) ---
# ¡IMPORTANTE! Esta es una estructura de ejemplo.
# Debes reemplazarla con la definición real de tu Hybrid_QNN,
# incluyendo la QuantumLayer que mencionas en tus documentos.
class Hybrid_QNN(nn.Module):
    def __init__(self):
        super(Hybrid_QNN, self).__init__()
        # Ejemplo de una capa clásica inicial
        self.cl_layer = nn.Linear(784, 4) # 784=28*28, 4 qubits
        
        # Aquí iría tu capa cuántica. Como no la tenemos, la simulamos con una capa clásica.
        # self.qlayer = QuantumLayer(4) # <-- Así se vería conceptualmente
        self.qlayer_placeholder = nn.Linear(4, 2) # Simula salida de 2 clases (e.g., dígitos 5 y 6)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Aplanar la imagen de entrada de [batch, 1, 28, 28] a [batch, 784]
        x = x.view(x.size(0), -1)
        x = self.cl_layer(x)
        
        # Pasar por la capa cuántica (o su sustituto)
        x = self.qlayer_placeholder(x)
        
        return self.softmax(x)

