# 1. Usar la imagen base de PyTorch Inference (CPU) que solicitaste
FROM public.ecr.aws/deep-learning-containers/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2-v1.43

# 2. Copiar tu código de inferencia. El directorio de trabajo de Lambda es /var/task.
COPY app.py /var/task/

# 3. Definir el punto de entrada de la función (handler)
# El formato es [nombre_archivo_sin_extension.nombre_funcion]
CMD ["app.handler"]