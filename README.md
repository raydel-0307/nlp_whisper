# Este repositorio contiene un algoritmo de Python para el fine-tuning del modelo GPT2. El código se basa en la librería Transformers de Hugging Face.

## Descripción del algoritmo

**El algoritmo entrena el modelo GPT2 con un conjunto de datos de texto personalizado. El objetivo es que el modelo aprenda patrones y estilos específicos de escritura a partir del conjunto de datos proporcionado.**

## Pre-requisitos

• Python 3.6 o superior
• Librerías de Python:
    • Transformers
    • datasets
    • pytorch

**Instalación**

bash```
pip install -r requirements.txt
```

## Configuración
* config.json

El archivo config.json contiene la configuración del proceso de fine-tuning. Se pueden modificar los parámetros de la configuración de acuerdo a las necesidades del usuario.

* Parámetros:

• model_name: Nombre del modelo GPT2 pre-entrenado.
• dataset: Lista de conjuntos de datos que se utilizarán para el entrenamiento.
    • type: Tipo del conjunto de datos (CSV, JSON, etc.).
    • filename: Nombre del archivo del conjunto de datos.
• prompt: Prompt inicial para el modelo.
• is_screen: Indica si se debe utilizar el modelo en modo de pantalla.
• max_lenght: Longitud del output
• prompt_type: Tipo de prompt, puede ser (audio,texto)
• whisper_model: Modelo que se va a usar (tiny, base, small, medium, large)

Nota: En caso de que se vaya a usar el modelo whisper!!! el prompt sería la ruta del archivo de audio

## Ejecución

1. Asegúrate de que el archivo config.json esté configurado correctamente.
2. Ejecuta el script train.py para que se realice la descarga del modelo
3. Ejecute el script fine_tune.py para que se realice el entrenamiendo del modelo
4. Ejecute el script main.py para realizar una generación


Ejemplo

Para ajustar el modelo GPT2 para generar recomendaciones de ritmos de rock, el archivo (JSON, CSV) podría contener las siguientes columnas:

json```
[
    {
        "text": "¿Quieres tener una piel radiante y llena de vitalidad? ¡Entonces nuestro nuevo producto es perfecto para ti!\n\nIntroducimos nuestro Serum Rejuvenecedor, diseñado específicamente para mujeres entre 25 y 40 años que desean combatir los signos del envejecimiento y lucir una piel joven y saludable.\n\nNuestra fórmula única está cargada de ingredientes naturales y potentes, como el ácido hialurónico y la vitamina C, que hidratan la piel, reducen las arrugas y aumentan la firmeza y elasticidad.\n\n¡No pierdas la oportunidad de probar este increíble producto! ¡Compra ahora y comienza a disfrutar de una piel increíblemente radiante! #SerumRejuvenecedor #PielJoven #AparienciaSaludable"
    },
    {
        "text": "¡Atención a todas las fashionistas! 👗 ¿Estás buscando un producto innovador y glamoroso para destacar en tus outfits?🌟 ¡Tenemos la solución perfecta para ti! Presentamos nuestro nuevo producto estrella: el collar ShineBright✨.\n\nEste collar elegante y moderno es el complemento ideal para cualquier look, ya sea casual o formal. Su diseño único y brillante hará que destaques en cualquier ocasión. ¡No pasarás desapercibida!\n\nPara todas las mujeres entre 18 y 35 años que buscan destacar su estilo con un toque de elegancia, el collar ShineBright es el accesorio imprescindible en tu colección.\n\nNo pierdas la oportunidad de ser una de las primeras en lucir este increíble collar. ¡Haz tu pedido ahora y comienza a brillar con ShineBright!✨ #ShineBright #EstiloÚnico #AccesorioImprescindible"
    },
    {
        "text": "🌟 ¡Atención a todas las amantes de la moda! 🌟\n\n¿Estás buscando un nuevo look para lucir esta temporada? ¡No busques más! Tenemos el producto perfecto para ti: ¡nuestros nuevos y exclusivos zapatos de diseño! 👠✨\n\nPara todas las mujeres entre 25 y 40 años que buscan estar a la moda y destacar con un estilo único, nuestros zapatos son la elección ideal. Confeccionados con los mejores materiales y diseñados por expertos en moda, te garantizamos que lucirás espectacular en cualquier ocasión.\n\nAdemás, con nuestra increíble promoción de lanzamiento, podrás conseguir un descuento del 20% en tu primera compra. ¡No te lo pierdas y renueva tu armario con nuestros zapatos de diseño!\n\n¡Visita nuestra tienda online y hazte con tus nuevos zapatos favoritos! 🛍️💖 #Moda #ZapatosDeDiseño #EstiloÚnico #Descuento #Fashionistas"
    }
]
```

Notas

• El rendimiento del modelo depende del conjunto de datos y la configuración del entrenamiento.
• Se recomienda ajustar los parámetros del entrenamiento y el conjunto de datos para obtener el mejor rendimiento del modelo.
• El modelo GPT2 es un modelo de lenguaje de gran tamaño, por lo que se requiere una gran cantidad de recursos computacionales para entrenarlo.