import os

variables_entorno = []

token_hf = os.getenv("TOKEN_HF")
model_segmentation_diarization = os.getenv("MODEL_SEGMENTATION_DIARIZATION")
variables_entorno.append(token_hf)
variables_entorno.append(model_segmentation_diarization)
# validar que el token no sea "" o None
if token_hf is None or token_hf == "":
    raise ValueError(
        "No se ha encontrado el token de Hugging Face en las variables de entorno."
    )


if __name__ == "__main__":
    pass
