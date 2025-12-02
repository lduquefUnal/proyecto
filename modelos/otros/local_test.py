import base64
import json
from PIL import Image
import os
import logging

# Importa las funciones de tu script de inferencia
from code.inference_quantum import model_fn, input_fn, predict_fn, output_fn

# Configura un logger básico para ver los mensajes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_local_test(image_path: str, model_directory: str = "."):
    """
    Simula el flujo de inferencia de SageMaker localmente.

    Args:
        image_path (str): Ruta a la imagen de prueba (ej: 'test_digit.png').
        model_directory (str): Directorio donde se encuentra 'model.pth'.
                               Por defecto es el directorio actual ('.').
    """
    logger.info("--- INICIANDO PRUEBA DE INFERENCIA LOCAL ---")

    # 1. Cargar el modelo (simula el inicio del endpoint)
    logger.info(f"Cargando modelo desde el directorio: {model_directory}")
    model_info = model_fn(model_directory)
    logger.info("Modelo cargado exitosamente.")

    # 2. Preparar la entrada (simula una petición HTTP)
    logger.info(f"Leyendo y codificando la imagen: {image_path}")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Codificar en base64 y crear el payload JSON
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    request_body = json.dumps({"input": image_b64})
    request_content_type = 'application/json'

    # 3. Deserializar la entrada con input_fn
    logger.info("Llamando a input_fn para procesar la petición...")
    image = input_fn(request_body, request_content_type)
    logger.info("input_fn completado.")

    # 4. Realizar la predicción con predict_fn
    logger.info("Llamando a predict_fn para obtener la predicción...")
    prediction = predict_fn(image, model_info)
    logger.info("predict_fn completado.")

    # 5. Serializar la salida con output_fn
    logger.info("Llamando a output_fn para formatear la respuesta...")
    response_content_type = 'application/json'
    final_response = output_fn(prediction, response_content_type)
    logger.info("output_fn completado.")

    # 6. Mostrar el resultado final
    logger.info("--- RESULTADO DE LA INFERENCIA ---")
    print(json.dumps(json.loads(final_response), indent=2))
    logger.info("--- PRUEBA FINALIZADA ---")

if __name__ == '__main__':
    # Obtener la ruta del directorio donde se encuentra este script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- CONFIGURA TU PRUEBA AQUÍ ---
    # Reemplaza 'test_digit.png' con la ruta a tu imagen de prueba.
    # La imagen debe ser un dígito en escala de grises, idealmente de 28x28 píxeles.
    # Construimos la ruta completa a la imagen
    test_image_file = os.path.join(script_dir, 'test_digit.png')

    logger.info(f"Ruta completa del script: {script_dir}")
    logger.info(f"Intentando cargar imagen desde: {test_image_file}")
    # El modelo ('model.pth') se encuentra en el mismo directorio que este script.
    model_dir = script_dir

    try:
        # El error puede ocurrir al cargar el modelo o al cargar la imagen.
        # El bloque try/except ahora es más específico.
        run_local_test(image_path=test_image_file, model_directory=model_dir)

    except FileNotFoundError as e:
        # Comprobamos si el error viene de no encontrar 'model.pth'
        if 'model.pth' in str(e):
            logger.error(f"Error: No se encontró el archivo del modelo 'model.pth' en el directorio '{model_dir}'.")
            logger.error("Por favor, renombra tu archivo de pesos del modelo (ej: 'mi_modelo.pth') a 'model.pth'.")
        else:
            logger.error(f"Error: El archivo de imagen '{test_image_file}' no fue encontrado. Detalles: {e}")
    except Exception as e:
        logger.error(f"Ocurrió un error durante la prueba: {e}", exc_info=True)
