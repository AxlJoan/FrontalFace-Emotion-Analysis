import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from threading import Thread
from collections import Counter
from pytube import YouTube
import os
import subprocess
import matplotlib.pyplot as plt

resultados = Counter()

def descargar_video(url):
    try:
        output_file = "video_youtube.mp4"
        command = ["yt-dlp", "-o", output_file, "-f", "mp4", url]
        subprocess.run(command, check=True)
        return output_file
    except Exception as e:
        print(f"Error al descargar el video: {e}")
        messagebox.showerror("Error", f"No se pudo descargar el video. {e}")
        return None

# Diccionario de traducción de emociones
emociones_traducidas = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Enojado",
    "fear": "Miedo",
    "surprise": "Sorpresa",
    "neutral": "Neutral",
    "disgust": "Disgusto",
}

def analizar_video(fuente, archivo_descargado=None):
    global resultados
    resultados.clear()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(fuente)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7)

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']

                # Traducir emoción detectada al español
                emocion_es = emociones_traducidas.get(emotion, emotion)
                resultados[emocion_es] += 1

                # Dibujar rectángulo y mostrar emoción traducida
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emocion_es, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error en análisis: {e}")

        cv2.imshow('Análisis de Emociones', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if archivo_descargado:  # Eliminar el archivo solo si fue descargado
        limpiar_archivo(archivo_descargado)

    mostrar_resumen()


def mostrar_resumen():
    # Crear resumen de emociones
    resumen = "\n".join([f"{emocion}: {cantidad}" for emocion, cantidad in resultados.items()])
    messagebox.showinfo("Resumen de Emociones Detectadas", resumen)

    # Traducir emociones para el gráfico
    emociones_es = {emociones_traducidas.get(emocion, emocion): cantidad for emocion, cantidad in resultados.items()}

    # Crear gráfico de pastel
    plt.figure(figsize=(8, 8))
    plt.pie(
        emociones_es.values(),
        labels=emociones_es.keys(),
        autopct='%1.1f%%',
        colors=plt.cm.Pastel1.colors,
        startangle=90,
    )
    plt.title("Distribución de Emociones Detectadas")
    plt.axis('equal')  # Asegura que el gráfico sea un círculo perfecto

    # Mostrar la gráfica
    plt.show()

def seleccionar_video():
    archivo = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi")])
    if archivo:
        Thread(target=analizar_video, args=(archivo,)).start()

def usar_camara():
    Thread(target=analizar_video, args=(0,)).start()

def procesar_youtube():
    enlace = simpledialog.askstring("Enlace de YouTube", "Introduce el enlace del video:")
    if enlace:
        ruta_video = descargar_video(enlace)
        if ruta_video:
            Thread(target=analizar_video, args=(ruta_video, ruta_video)).start()  # Pasamos la ruta del archivo descargado
        else:
            print("No se pudo descargar el video.")

def limpiar_archivo(ruta):
    if os.path.exists(ruta):
        os.remove(ruta)
        print(f"Archivo {ruta} eliminado.")

# Crear la interfaz
ventana = tk.Tk()
ventana.title("Analizador de Emociones")

tk.Label(ventana, text="Analizador de Emociones en Videos", font=("Arial", 16)).pack(pady=10)
tk.Button(ventana, text="Usar Cámara", command=usar_camara, width=20).pack(pady=5)
tk.Button(ventana, text="Seleccionar Video", command=seleccionar_video, width=20).pack(pady=5)
tk.Button(ventana, text="Procesar YouTube", command=procesar_youtube, width=20).pack(pady=5)
tk.Button(ventana, text="Salir", command=ventana.quit, width=20).pack(pady=20)

ventana.mainloop()
