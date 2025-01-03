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
from mss import mss
from PIL import Image, ImageTk
import numpy as np

resultados = Counter()

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

class Analizador:
    def __init__(self):
        self.resultados = Counter()  # Usamos un contador específico para cada instancia
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self.running = False
        self.ventana = tk.Tk()
        self.ventana.title("Analizador de Emociones")
        self.ventana.geometry("640x480")
        self.ventana.resizable(True, True)

        # Lienzo para mostrar el video
        self.lienzo = tk.Canvas(self.ventana, width=640, height=360, bg="black")
        self.lienzo.pack(pady=10, fill=tk.BOTH, expand=True)

        # Botones
        botones_frame = tk.Frame(self.ventana)
        botones_frame.pack(pady=10, fill=tk.X, padx=5)
        tk.Button(botones_frame, text="Usar Cámara", command=self.usar_camara, width=20).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        tk.Button(botones_frame, text="Seleccionar Video", command=self.seleccionar_video, width=20).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(botones_frame, text="Procesar YouTube", command=self.procesar_youtube, width=20).grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        tk.Button(botones_frame, text="Analizar Pantalla", command=self.capturar_pantalla, width=20).grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        tk.Button(botones_frame, text="Detener", command=self.detener, width=40, bg="red", fg="white").grid(row=1, column=0, columnspan=4, padx=5, pady=5)

    def capturar_pantalla(self):
        """Iniciar captura de pantalla y análisis de emociones en tiempo real"""
        self.resultados.clear()  # Limpiar los resultados previos
        self.running = True
        self._captura_pantalla()  # Llamamos a la función de captura

    def _captura_pantalla(self):
        """Función de captura de pantalla que se ejecuta en un ciclo"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        with mss() as sct:
            monitor = sct.monitors[0]  # Captura toda la pantalla

            # Captura un fotograma de la pantalla
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convertir a BGR

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7)

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]

                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    emocion_es = emociones_traducidas.get(emotion, emotion)
                    self.resultados[emocion_es] += 1

                    # Dibujar rectángulo y mostrar emoción traducida
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emocion_es, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error en análisis: {e}")

            # Redimensionamos el frame a las dimensiones de la ventana
            w, h = self.lienzo.winfo_width(), self.lienzo.winfo_height()
            resized_frame = cv2.resize(frame, (w, h))

            # Mostrar la captura redimensionada en la ventana de Tkinter
            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.lienzo.create_image(0, 0, anchor="nw", image=img)
            self.lienzo.image = img  # Retiene la referencia de la imagen para evitar que se libere

            # Usar after para mantener el ciclo de captura
            if self.running:
                self.ventana.after(10, self._captura_pantalla)

            self.ventana.update_idletasks()  # Refrescar la interfaz para que se actualicen los cambios

    def mostrar_resumen(self):
        """Mostrar el resumen de las emociones"""
        emociones_es = {emociones_traducidas.get(emocion, emocion): cantidad for emocion, cantidad in self.resultados.items()}

        plt.figure(figsize=(8, 8))
        plt.pie(
            emociones_es.values(),
            labels=emociones_es.keys(),
            autopct='%1.1f%%',
            colors=plt.cm.Pastel1.colors,
            startangle=90,
        )
        plt.title("Distribución de Emociones Detectadas")
        plt.axis('equal')
        plt.show()

    def procesar_youtube(self):
        enlace = simpledialog.askstring("Enlace de YouTube", "Introduce el enlace del video:")
        if enlace:
            ruta_video = self.descargar_video(enlace)
            if ruta_video:
                Thread(target=self.analizar_video, args=(ruta_video, ruta_video)).start()
            else:
                print("No se pudo descargar el video.")

    def descargar_video(self, url):
        try:
            output_file = "video_youtube.mp4"
            command = ["yt-dlp", "-o", output_file, "-f", "mp4", url]
            subprocess.run(command, check=True)
            return output_file
        except Exception as e:
            print(f"Error al descargar el video: {e}")
            return None

    def analizar_video(self, fuente, archivo_descargado=None):
        """Analiza un video y detecta emociones en tiempo real"""
        self.resultados.clear()  # Limpiar los resultados antes de iniciar el análisis
        cap = cv2.VideoCapture(fuente)
        self.cap = cap  # Guardamos la referencia del VideoCapture para cerrarlo después

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 360))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=7)

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    emocion_es = emociones_traducidas.get(emotion, emotion)
                    self.resultados[emocion_es] += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emocion_es, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error en análisis: {e}")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.lienzo.create_image(0, 0, anchor="nw", image=img)
            self.lienzo.image = img  # Retiene referencia para evitar que se libere
            self.ventana.update_idletasks()

        cap.release()

        if archivo_descargado:
            self.limpiar_archivo(archivo_descargado)

        self.mostrar_resumen()  # Llamar a mostrar_resumen para generar la gráfica


    def seleccionar_video(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi")])
        if archivo:
            Thread(target=self.analizar_video, args=(archivo,)).start()

    def detener(self):
        self.running = False  # Detener análisis de pantalla
        self.mostrar_resumen()  # Mostrar el resumen de emociones

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()  # Detener el video
            cv2.destroyAllWindows()  # Cerrar cualquier ventana de OpenCV

    def limpiar_archivo(self, archivo_descargado):
        if os.path.exists(archivo_descargado):
            os.remove(archivo_descargado)
            print("Archivo temporal eliminado.")

    def usar_camara(self):
        """Inicia la captura desde la cámara."""
        Thread(target=self.analizar_video, args=(0,)).start()  # El 0 es el índice de la cámara

    def iniciar(self):
        self.running = True
        self.ventana.mainloop()

# Crear y ejecutar el analizador
analizador = Analizador()
analizador.iniciar()
