# Proocessos

## TO DO

### Definir Circuito Cuántico

  - priority: high
  - workload: Hard
  - defaultExpanded: false
    ```md
    Implementar la clase QuantumFunction o equivalente en el notebook que defina el kernel cuántico (e.g., usando ry y rx en cudaq.kernel) y las funciones forward y backward para el cálculo de expectativas y gradientes.
    ```

### Crear Capa Cuántica (QLayer)

  - priority: medium
  - workload: Normal
  - defaultExpanded: false
    ```md
    Desarrollar la clase QuantumLayer que encapsula la lógica del circuito cuántico en una capa compatible con PyTorch (nn.Module).
    ```

### Definir Modelo HQNN Completo

  - priority: medium
  - workload: Normal
  - defaultExpanded: false
    ```md
    Implementar la clase Hybrid_QNN que combine las capas clásicas (nn.Linear, nn.Dropout, torch.relu) con la QuantumLayer.
    ```

### Entrenamiento Básico

  - priority: medium
  - workload: Normal
  - defaultExpanded: false
    ```md
    Configurar el optimizador y la función de pérdida (e.g., optim.Adadelta, nn.BCELoss) e implementar el bucle de entrenamiento para un número inicial de épocas.
    ```

### Evaluación y Métricas

  - defaultExpanded: false
    ```md
    Implementar la función de cálculo de precisión (accuracy_score) y añadir la lógica para registrar los costos y la precisión de entrenamiento y prueba por época.
    ```

## In Progres

### Preparación de Datos (MNIST)

  - priority: medium
  - workload: Normal
  - defaultExpanded: false
    ```md
    Implementar la función prepare_data para cargar, filtrar y normalizar el dataset (e.g., MNIST dígitos 5 y 6), incluyendo la división en sets de entrenamiento y prueba.
    ```

## DONE

### Setup Entorno Cuántico

  - priority: medium
  - workload: Easy
  - defaultExpanded: false
    ```md
    Instalar todas las librerías necesarias (CUDA-Q, PyTorch, etc.) y configurar el entorno Python para el proyecto.
    ```

