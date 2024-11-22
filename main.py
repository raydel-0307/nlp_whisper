import json
import iamodels


def fuctions_execute(config_json_path: str):
    # Leer el archivo de configuración
    with open(config_json_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    #Ruta del proyecto
    ruta = config["proyect"]
    with open(f"{ruta}/{config_json_path}", 'r', encoding='utf-8') as file:
        config = json.load(file)
        
    # Usar los valores del archivo JSON
    prompt = config["prompt"]
    is_screen = config["is_screen"]
    max_length = config["max_length"]
    temperature = config["temperature"]
    top_k = config["top_k"]
    top_p = config["top_p"]

    prompt_type = config["prompt_type"]
    
    if prompt_type=="audio":
        whisper_model = config["whisper_model"]
    else:
        whisper_model = None

    # Llamar al modelo y mostrar los resultados
    iamodels.generate(ruta, prompt,prompt_type,whisper_model,is_screen,max_length,temperature,top_k,top_p)


def main():

    config_json_path = "config.json"

    result = fuctions_execute(config_json_path)

if __name__ == "__main__":
    main()
