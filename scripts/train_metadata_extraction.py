"""
Script para la extracción de metadatos técnicos de videos de entrenamiento.

Este script procesa un directorio de videos MP4, extrae información técnica relevante
(duración, resolución, FPS, audio y bitrate) utilizando la herramienta del sistema ffprobe 
y guarda los resultados en un archivo CSV para su posterior análisis en modelos de ML.
"""

import subprocess # Módulo para ejecutar comandos del sistema (como ffprobe) desde Python.
import json       # Módulo para trabajar con datos en formato JSON, que es cómo ffprobe devolverá la información.
import csv        # Módulo para leer y escribir archivos CSV (Comma Separated Values), donde guardaremos los datos tabulares.
import os         # Módulo para interactuar con el sistema operativo (rutas de archivos, comprobar si existen directorios, etc.).
import argparse   # Módulo para manejar argumentos pasados por terminal (ej. --input folder_path).
import logging    # Módulo para registrar mensajes (logs) en lugar de usar simples 'print', útil para saber qué pasa durante la ejecución.
from typing import Optional, Dict, Any # Herramientas para indicar el tipo de dato que usan y devuelven las variables (ayuda al autocompletado y a leer el código).
from tqdm import tqdm # Módulo para mostrar una barra de progreso visual en la consola mientras procesamos los videos.

# Configuración de logs
# Configuramos cómo queremos que se vean los mensajes de información/error en la consola.
# En este caso, mostraremos la fecha/hora, el nivel de importancia (INFO, ERROR) y el mensaje.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """
    Extrae metadatos técnicos importantes de un archivo de video usando la herramienta externa 'ffprobe'.

    Args:
        video_path (str): Ruta completa al archivo de video que queremos analizar.

    Returns:
        Dict[str, Any]: Un diccionario con las características del video (metadatos). 
                        En caso de que el video falle o no tenga algún dato, ese campo mantendrá su valor por defecto.
    """
    # Inicializamos un diccionario con valores nulos (None) o ceros.
    # Si ffprobe falla o el video no tiene los datos, estos son los valores de seguridad que se devolverán.
    metadata = {
        'duration': None,
        'width': None,
        'height': None,
        'fps': None,
        'has_audio': 0,
        'bitrate': None
    }

    try:
        # Se define el comando de terminal que queremos ejecutar. 
        #
        # ¿Por qué se pasa como una lista y no como un solo string continuo (ej. "ffprobe -v quiet ...")?
        # Porque el módulo 'subprocess' es más seguro si se le pasa una lista. De esta forma el sistema
        # operativo sabe claramente qué elemento es el programa principal ('ffprobe'), y delimita perfectamente 
        # cuáles son sus argumentos. Esto evita errores terribles si, por ejemplo, `video_path` tuviera 
        # espacios en su nombre de archivo (ej. "mi video.mp4").
        cmd = [
            'ffprobe',                # El programa que vamos a ejecutar (debe estar instalado en el sistema).
            '-v', 'quiet',            # '-v quiet' le dice a ffprobe que sea silencioso y no imprima su logo inicial o en la consola que molestarían al leer tras procesar.
            '-print_format', 'json',  # Queremos que ffprobe devuelva la información de metadatos estrictamente en formato JSON, fácil de transformar en diccionario por Python.
            '-show_format',           # Pide información general del contenedor del video (duración total, tamaño en disco, bitrate global).
            '-show_streams',          # Pide información detallada por separado de cada "stream" o pista (pista de video, pista de audio, subtítulos).
            video_path                # El último argumento obligatorio de ffprobe es la ruta del archivo que va a ser inspeccionado.
        ]
        
        # check_output ejecuta el cmd (lista de texto) en la consola silenciosamente y guarda lo que el programa imprime de vuelta en su salida estándar.
        output = subprocess.check_output(cmd)
        
        # 'output' es texto crudo con formato JSON. Usamos json.loads() (load string) para convertir ese texto en un diccionario de Python.
        info = json.loads(output)
        
        # --- 1. Información de formato general ---
        # Extraemos la clave 'format' del diccionario. Si no existe usamos un diccionario {} vacío por seguridad.
        fmt = info.get('format', {})
        # Extraemos la duración y el bitrate (cantidad de datos por segundo).
        # Los convertimos a números en Python (float para duración porque tiene decimales, int para bitrate).
        metadata['duration'] = float(fmt.get('duration', 0))
        metadata['bitrate'] = int(fmt.get('bit_rate', 0))
        
        # --- 2. Información de streams (pistas individualizadas) ---
        # Un video suele tener una o varias pistas de imagen (video) y una o varias de sonido (audio).
        streams = info.get('streams', [])
        
        # Buscamos el stream encargado de la imagen ('codec_type' == 'video').
        # La función nativa 'next()' itera sobre un bucle y te da el "siguiente" (o el primer) elemento que cumpla la condición.
        # Aquí pasamos un iterador (s for s in streams si el codec_type es video). Si no entra ninguno, devuelve el "{}" de la derecha.
        video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
        
        # Hacemos lo mismo para el 'audio'. Si no hay pista de audio, devolverá None en lugar de un diccionario vacío.
        audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), None)
        
        # Si nuestra búsqueda del stream de video nos dio contenido real...
        if video_stream:
            metadata['width'] = int(video_stream.get('width', 0))
            metadata['height'] = int(video_stream.get('height', 0))
            
            # --- Cálculo de FPS (Frames Per Second / Fotogramas Por Segundo) ---
            # 'r_frame_rate' suele devolver fracciones como cadena de texto, ej. "30000/1001" (para 29.97 fps) o "24/1".
            # Hacemos un split('/') para convertir "30000/1001" en la lista ["30000", "1001"].
            fps_parts = video_stream.get('r_frame_rate', '0/1').split('/')
            
            # Verificamos que se haya dividido exactamente en 2 partes (numerador y denominador) y que no vayamos a dividir por 0.
            if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                # Dividimos numerador por denominador y redondeamos a 2 decimales usando round().
                metadata['fps'] = round(int(fps_parts[0]) / int(fps_parts[1]), 2)
            else:
                metadata['fps'] = 0
                
        # Si 'audio_stream' tuvo éxito al hacer su "next()", será diferente de None (por lo que guardará un 1 en nuestro diccionario, de lo contrario un 0).
        metadata['has_audio'] = 1 if audio_stream else 0
        
    except (subprocess.CalledProcessError, ValueError, KeyError, ZeroDivisionError) as e:
        # El bloque "try / except" atrapa cualquier error imprevisto (ej. archivo corrupto, dividir por 0 o fallo de permisos) y evita que
        # TODO el script que va procesando 1.000 videos, por ejemplo, detenga su flujo completo solo por 1 video malo.
        # Simplemente guardamos un error log del video en concreto notificándonoslo.
        logging.error(f"Error procesando {video_path}: {e}")
    
    # Devuelve el diccionario tanto si salió bien o como si saltó el error y se quedó en sus valores None basales.
    return metadata

def process_videos(video_dir: str, output_csv: str) -> None:
    """
    Recorre todos los videos MP4 de una carpeta, extrae su información usando la función individual
    get_video_metadata y guarda los resultados ordenadamente en el archivo CSV dado.

    Args:
        video_dir (str): Directorio donde están los videos físicamente.
        output_csv (str): Ruta completa donde queremos guardar nuestro archivo .csv al construirlo.
    """
    # Primero, comprobamos que la carpeta que nos han pasado realmente existe en el disco duro.
    if not os.path.exists(video_dir):
        logging.error(f"El directorio de videos no existe: {video_dir}")
        return

    # Usamos list comprehension para crear una lista solo con los archivos que terminan en '.mp4' ignorando mayúsculas/minúsculas.
    # os.listdir() devuelve TODOS los archivos que encuentre indiscriminadamente de manera alfabética pero plana. 
    files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    logging.info(f"Encontrados {len(files)} archivos .mp4 en {video_dir}")

    # Abrimos (o creamos) un archivo CSV en modo escritura de datos ('w' == write).
    # Usar el bloque 'with' asegura que el sistema operativo cerrará correctamente y liberará el archivo al 
    # terminar su iteración interna aunque salte algún error por el camino (muy vital a nivel de memoria).
    # 'newline=\'\'' previene la inyección de líneas dobles fantasma en sistemas como Windows que usan \r\n de salto de línea en ficheros.
    with open(output_csv, 'w', newline='', encoding='utf-8') as out:
        # El csv.writer facilita insertar listas (celdas separadas por comas) al archivo final
        writer = csv.writer(out)
        
        # Escribimos nuestra primera celda de cabeceras en el excel/CSV directamente
        writer.writerow(['Id', 'duration', 'width', 'height', 'fps', 'has_audio', 'bitrate'])
        
        # Empezamos el bucle. 'tqdm(files)' "envuelve" la lista 'files' y sirve para que nuestro script nos
        # dibuje e informe una preciosa barra de carga bonita por la consola mientra vamos del %0 al %100 de ficheros.
        for f in tqdm(files, desc="Extrayendo metadatos"):
            # Obtenemos el "ID único" del video partiendo la extensión original de éste. Ejemplo: "video_42.mp4" -> separamos solo en "video_42"
            vid_id = os.path.splitext(f)[0]
            
            # Formamos de manera limpia, independientemente del SO (Linux/Win/Mac), la ruta absoluta final del video en base a su carpeta y nombre propio.
            path = os.path.join(video_dir, f)
            
            # Llamamos a nuestra función principal delegándole la ruta específica del iterador en el bucle en este mismo instante.
            meta = get_video_metadata(path)
            
            # Escribimos los metadatos devueltos como una nueva línea para el final (append literal de writerow) de nuestro documento CSV.
            writer.writerow([
                vid_id, 
                meta['duration'], 
                meta['width'], 
                meta['height'], 
                meta['fps'], 
                meta['has_audio'], 
                meta['bitrate']
            ])

    # Cuando salimos del bloque "with", el log nos avisa de lo que ocurrió y del estado.
    logging.info(f"Extracción completada. Resultados guardados en: {output_csv}")

def main():
    """
    Punto de entrada principal del script. 
    Usamos el módulo 'argparse' para crear una interfaz de comandos para línea (CLI).
    Esto es una buena práctica y permite que modifiques rutas (inputs/outputs) cuando uses tu terminal y llames al script,
    ejecutando en consola: $ python script.py --input /mi/ruta_diferente --output /mia/otra
    sin tener que estar reabriendo este arhivo python usando 'VS Code' y cambiar sus variables predeterminadas cada vez.
    """
    # Creamos un parser y una descripción del comando ayuda 'help' (-h).
    parser = argparse.ArgumentParser(description="Extractor de metadatos de video para SnapUGC.")
    
    # Añadimos un argumento que el usuario elegirá llamar "--input". Añadimos su configuración de default.
    parser.add_argument(
        "--input", 
        default="/media/5tbraid/data/martugue/SnapUGC/train_videos/",
        help="Directorio de entrada de videos"
    )
    
    # Añadimos argumentación para "--output".
    parser.add_argument(
        "--output", 
        default="/media/5tbraid/data/martugue/SnapUGC/train_metadata.csv",
        help="Archivo CSV de salida"
    )
    
    # El parse_args() es la función encarga de "leer tu teclado/consola" lo que el usuario ha solicitado 
    # en la ejecución, y crea un objeto "args" donde:
    #   'args.input' guarda la ruta input
    #   'args.output' guarda la ruta output.
    args = parser.parse_args()
    
    # Llamamos a la lógica principal.
    process_videos(args.input, args.output)

# En Python, "__name__" es una variable especial. 
# Si el script se ejecuta de frente con "python script.py", Python le asgina al script de forma interna el nombre especial "__main__".
# Si este script fuese empaquetado como módulo a importar en otro script con 'import script', su __name__ sería distinto a '__main__' y las funciones solo se cargarían, pero sin ser ejecutadas por sorpresa en esta línea:
if __name__ == "__main__":
    main()