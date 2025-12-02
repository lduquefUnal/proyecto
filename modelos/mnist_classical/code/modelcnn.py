import torch
import torch.nn as nn
import numpy as np

# --- Importaciones Cuánticas ---
# Asegúrate de que cuda-quantum esté instalado en el entorno de SageMaker
# (puedes añadir 'cuda-quantum' a tu requirements.txt si es necesario)
try:
    import cudaq
    from cudaq import spin
except ImportError:
    print("Advertencia: No se pudo importar cudaq. La parte cuántica del modelo no funcionará.")

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

# --- Modelo Híbrido Cuántico-Clásico (HQNN) - Extraído de red_hibrida.ipynb ---

# 1. Definición del Kernel Cuántico y Hamiltoniano
# Estos son globales para el módulo, ya que no dependen de la instancia de la clase.
n_qubits = 4

try:
    kernel, features = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)

    # Codificación de características con rotaciones RY
    for i in range(n_qubits):
        kernel.ry(features[i], qubits[i])

    # Entrelazamiento con compuertas CNOT
    for i in range(n_qubits - 1):
        kernel.cx(qubits[i], qubits[i + 1])

    # Define el observable a medir (Suma de operadores Z)
    hamiltonian = sum(spin.z(i) for i in range(n_qubits))
except (NameError, AttributeError):
    # Si cudaq no se pudo importar, estas variables no se crearán.
    # Esto permite que el archivo se importe en entornos sin cudaq, aunque no se pueda ejecutar.
    kernel, hamiltonian = None, None

# 2. Función de Autograd para la Capa Cuántica
class QuantumFunction(torch.autograd.Function):
    """Función de autograd para ejecutar el circuito y calcular gradientes."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        """Pase hacia adelante: ejecuta el circuito cuántico."""
        if kernel is None or hamiltonian is None:
            raise RuntimeError("El kernel cuántico no está inicializado. ¿Se importó cudaq correctamente?")
        
        ctx.save_for_backward(x)
        exp_vals_list = []
        for i in range(x.shape[0]):
            result_future = cudaq.observe_async(kernel, hamiltonian, x[i])
            exp_vals_list.append(result_future.get().expectation())
        
        output = torch.tensor(exp_vals_list, device=x.device, dtype=x.dtype).reshape(-1, 1)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Pase hacia atrás: calcula gradientes con parameter-shift."""
        x, = ctx.saved_tensors
        batch_size, n_params = x.shape
        gradients = torch.zeros_like(x)

        for i in range(n_params):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[:, i] += np.pi / 2.0
            x_minus[:, i] -= np.pi / 2.0
            
            exp_vals_plus_list = []
            exp_vals_minus_list = []
            for j in range(batch_size):
                future_plus = cudaq.observe_async(kernel, hamiltonian, x_plus[j])
                future_minus = cudaq.observe_async(kernel, hamiltonian, x_minus[j])
                exp_vals_plus_list.append(future_plus.get().expectation())
                exp_vals_minus_list.append(future_minus.get().expectation())
            
            exp_vals_plus = torch.tensor(exp_vals_plus_list, device=x.device, dtype=x.dtype)
            exp_vals_minus = torch.tensor(exp_vals_minus_list, device=x.device, dtype=x.dtype)
            
            gradient_component = 0.5 * (exp_vals_plus - exp_vals_minus)
            gradients[:, i] = (gradient_component * grad_output).sum(dim=1)
            
        return gradients

# 3. Módulo de PyTorch para la Capa Cuántica
class QuantumLayer(nn.Module):
    """Capa que encapsula la función cuántica."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return QuantumFunction.apply(x)

# 4. Clase del Modelo Híbrido (antes HybridCNN)
class Hybrid_QNN(nn.Module):
    """Red Neuronal Híbrida: CNN Clásica + Capa Cuántica"""
    def __init__(self, n_qubits: int = 4):
        super(Hybrid_QNN, self).__init__()
        self.n_qubits = n_qubits
        
        # Parte clásica: Convoluciones
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.flatten = nn.Flatten()
        self.pre_quantum_fc = nn.Linear(64 * 5 * 5, self.n_qubits)
        self.quantum_layer = QuantumLayer()
        self.post_quantum_fc = nn.Linear(1, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.pre_quantum_fc(x)
        x = torch.sigmoid(x) * np.pi
        x = self.quantum_layer(x)
        logits = self.post_quantum_fc(x)
        # Aplicamos softmax para que la salida sean probabilidades, como espera inference.py
        return self.softmax(logits)
