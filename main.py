from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import shutil
import os
import uuid
import mysql.connector
from typing import Optional

# Conexión a la base de datos
conn = mysql.connector.connect(
    host="mainline.proxy.rlwy.net",
    user="root",
    password="FYffArrABIHbQZGhepeODICIyjFnxMzD",
    database="railway",
    port=52052,
)
cursor = conn.cursor()

# Crear tabla detecciones si no existe
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS detecciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre_archivo VARCHAR(255) NOT NULL,
    clase_detectada VARCHAR(50),
    ruta_imagen TEXT,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
)
conn.commit()

# Conexión 2 - martillos
conn_martillos = mysql.connector.connect(
    host="shinkansen.proxy.rlwy.net",
    user="root",
    password="tKpYauIADsPJdjxAEniuqXLOfQBdiVUL",
    database="railway",
    port=54369,
)
cursor_martillos = conn_martillos.cursor()

# Crear tabla si no existe
cursor_martillos.execute(
    """
CREATE TABLE IF NOT EXISTS martillos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    cantidad INT NOT NULL,
    precio DECIMAL(10, 2) NOT NULL
)
"""
)
conn_martillos.commit()


app = FastAPI(
    title="API de Detección de Objetos",
    description="Sube una imagen para detectar objetos con YOLO y descarga la imagen procesada.",
    version="1.0.0",
)

# Cargar modelo
modelo = YOLO("Herramientas_model/entrenamiento_herramientas/weights/best.pt")

# Crear directorio de salida si no existe
os.makedirs("imagenes_salida", exist_ok=True)


@app.post("/detectar/", response_class=FileResponse)
async def detectar_objeto(imagen: UploadFile = File(...)):
    id_img = str(uuid.uuid4())
    ruta_entrada = f"temp_{id_img}.jpg"
    ruta_salida = f"imagenes_salida/predict_{id_img}.jpg"

    with open(ruta_entrada, "wb") as f:
        shutil.copyfileobj(imagen.file, f)

    resultados = modelo.predict(
        source=ruta_entrada,
        save=True,
        project="imagenes_salida",
        name=f"predict_{id_img}",
        exist_ok=True,
    )

    # Obtener clase detectada (solo la primera si hay varias)
    clase_detectada: Optional[str] = None
    if resultados and resultados[0].boxes and resultados[0].boxes.cls.numel() > 0:
        clase_detectada = float(resultados[0].boxes.cls[0])
    else:
        clase_detectada = "sin detección"

    # Buscar la imagen procesada
    carpeta_prediccion = f"imagenes_salida/predict_{id_img}"
    archivos_generados = os.listdir(carpeta_prediccion)
    if archivos_generados:
        imagen_procesada = os.path.join(carpeta_prediccion, archivos_generados[0])
    else:
        return {"error": "No se generó ninguna imagen"}

    # Guardar en base de datos
    cursor.execute(
        "INSERT INTO detecciones (nombre_archivo, clase_detectada, ruta_imagen) VALUES (%s, %s, %s)",
        (imagen.filename, str(clase_detectada), imagen_procesada),
    )
    conn.commit()

    return FileResponse(
        path=imagen_procesada,
        filename=f"resultado_{imagen.filename}",
        media_type="image/jpeg",
    )


@app.get("/ultima-imagen", response_class=FileResponse)
def obtener_ultima_imagen():
    cursor.execute("SELECT ruta_imagen FROM detecciones ORDER BY id DESC LIMIT 1")
    resultado = cursor.fetchone()

    if not resultado:
        return {"error": "No hay imágenes registradas"}

    ruta_imagen = resultado[0]

    if not os.path.exists(ruta_imagen):
        return {"error": "La imagen no se encuentra en el sistema"}

    return FileResponse(
        path=ruta_imagen, media_type="image/jpeg", filename="ultima_deteccion.jpg"
    )


@app.get("/martillos")
def obtener_martillos():
    cursor_martillos.execute("SELECT nombre, cantidad, precio FROM martillos")
    martillos = cursor_martillos.fetchall()
    return JSONResponse(
        [
            {"nombre": nombre, "cantidad": cantidad, "precio": float(precio)}
            for (nombre, cantidad, precio) in martillos
        ]
    )


@app.get("/feed-imagenes")
def obtener_urls_imagenes(request: Request):
    cursor.execute("SELECT ruta_imagen FROM detecciones ORDER BY id DESC")
    resultados = cursor.fetchall()

    base_url = str(request.base_url).rstrip("/")  # Ej: http://127.0.0.1:8000
    urls = []

    for (ruta,) in resultados:
        nombre_archivo = os.path.basename(ruta)
        urls.append(f"{base_url}/imagen-feed/{nombre_archivo}")

    return urls


@app.get("/imagen-feed/{nombre}", response_class=FileResponse)
def servir_imagen_feed(nombre: str):
    ruta_imagen = f"imagenes_salida/{nombre}"

    # Si la imagen fue generada dentro de una subcarpeta tipo predict_xxx
    if not os.path.exists(ruta_imagen):
        for carpeta in os.listdir("imagenes_salida"):
            posible_ruta = os.path.join("imagenes_salida", carpeta, nombre)
            if os.path.exists(posible_ruta):
                ruta_imagen = posible_ruta
                break
        else:
            return JSONResponse(
                content={"error": "Imagen no encontrada"}, status_code=404
            )

    return FileResponse(path=ruta_imagen, media_type="image/jpeg", filename=nombre)
