import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import filedialog, Toplevel
from threading import Thread
from collections import Counter
import os
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import warnings

# Suprimir las advertencias específicas
warnings.filterwarnings("ignore", message="Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.")

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

# Ruta a la base de datos de clientes
clientes_path = "Clientes"

class Analizador:
    def __init__(self):
        self.resultados_clientes = Counter()
        self.resultados_general = Counter()
        self.rostros_codificados = []
        self.nombres_clientes = []
        self.running = False
        self.cap = None  

        # Cargar banco de clientes
        self.cargar_banco_clientes()

        # Configuración de la interfaz gráfica
        self.ventana = tk.Tk()
        self.ventana.title("Analizador de Emociones y Rostros")
        self.ventana.geometry("900x1000")  

        # Frame para el video
        video_frame = tk.Frame(self.ventana)
        video_frame.pack(side=tk.TOP, pady=10)

        # Lienzo para mostrar el video
        self.lienzo = tk.Canvas(video_frame, width=640, height=360, bg="black")
        self.lienzo.pack()

        # Frame para los botones
        botones_frame = tk.Frame(self.ventana)
        botones_frame.pack(side=tk.TOP, pady=10)

        # Botones
        tk.Button(botones_frame, text="Usar Cámara", command=self.usar_camara, width=20).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(botones_frame, text="Seleccionar Video", command=self.seleccionar_video, width=20).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(botones_frame, text="Detener", command=self.detener, width=20, bg="red", fg="white").grid(row=0, column=2, padx=5, pady=5)

        # Frame para las gráficas
        graficas_frame = tk.Frame(self.ventana)
        graficas_frame.pack(side=tk.TOP, pady=10, fill=tk.BOTH, expand=True)

        # Figura para las gráficas
        self.figura, (self.ax_clientes, self.ax_general) = plt.subplots(1, 2, figsize=(10, 5))
        self.figura.suptitle("Distribución de Emociones Detectadas")

        self.ax_clientes.set_title("Cliente")
        self.ax_general.set_title("General")
        self.ax_clientes.axis("equal")
        self.ax_general.axis("equal")

        # Canvas para incrustar la gráfica en la interfaz
        self.canvas_grafico = FigureCanvasTkAgg(self.figura, master=graficas_frame)
        self.canvas_grafico.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def cargar_banco_clientes(self):
        """Cargar las imágenes de los clientes desde las subcarpetas y calcular sus codificaciones."""
        for carpeta_cliente in os.listdir(clientes_path):
            ruta_carpeta = os.path.join(clientes_path, carpeta_cliente)
            if os.path.isdir(ruta_carpeta):  
                for archivo in os.listdir(ruta_carpeta):
                    if archivo.endswith(('jpg', 'png', 'jpeg')):
                        ruta_imagen = os.path.join(ruta_carpeta, archivo)
                        imagen = face_recognition.load_image_file(ruta_imagen)
                        codificacion = face_recognition.face_encodings(imagen)
                        if codificacion:
                            self.rostros_codificados.append(codificacion[0])
                            self.nombres_clientes.append(carpeta_cliente)  

    def analizar_video(self, fuente):
        """Analiza un video y detecta emociones y rostros en tiempo real."""
        self.resultados_clientes.clear()
        self.resultados_general.clear()
        self.running = True  
        self.cap = cv2.VideoCapture(fuente)

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convertir a RGB antes de procesar (evita el tinte azul)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ubicaciones_rostros = face_recognition.face_locations(rgb_frame)
            codificaciones_rostros = face_recognition.face_encodings(rgb_frame, ubicaciones_rostros)

            for codificacion, (top, right, bottom, left) in zip(codificaciones_rostros, ubicaciones_rostros):
                # Identificar el cliente
                coincidencias = face_recognition.compare_faces(self.rostros_codificados, codificacion)
                nombre = "Desconocido"

                if True in coincidencias:
                    indices = [i for i, coinc in enumerate(coincidencias) if coinc]
                    mejor_indice = indices[np.argmin(face_recognition.face_distance([self.rostros_codificados[i] for i in indices], codificacion))]
                    nombre = self.nombres_clientes[mejor_indice]

                roi = rgb_frame[top:bottom, left:right]

                # Detectar emoción
                try:
                    result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
                    emocion = emociones_traducidas.get(result[0]['dominant_emotion'], result[0]['dominant_emotion'])

                    # Registrar emoción en el grupo correspondiente
                    if nombre == "Desconocido":
                        self.resultados_general[emocion] += 1
                    else:
                        self.resultados_clientes[emocion] += 1

                    # Mostrar en pantalla
                    cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                    # Crear un fondo blanco detrás del texto
                    text = f"{nombre}: {emocion}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_width, text_height = text_size

                    # Coordenadas para el rectángulo blanco
                    rect_x = left
                    rect_y = top - 30
                    rect_width = text_width + 10
                    rect_height = text_height + 10

                    # Dibujar el rectángulo blanco
                    cv2.rectangle(rgb_frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 255, 255), -1)

                    # Dibujar el texto en negro sobre el fondo blanco
                    cv2.putText(rgb_frame, text, (left + 5, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                except Exception as e:
                    print(f"Error al analizar emoción: {e}")

            # Ajustar el video al tamaño del lienzo
            frame_resized = cv2.resize(rgb_frame, (640, 360))
            img = ImageTk.PhotoImage(Image.fromarray(frame_resized))
            self.lienzo.create_image(0, 0, anchor="nw", image=img)
            self.lienzo.image = img
            self.actualizar_grafico()
            self.ventana.update_idletasks()

        self.cap.release()

    def detener(self):
        """Detiene el análisis en curso y muestra las gráficas en una ventana emergente."""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.actualizar_grafico()
        self.mostrar_resumen()

    def mostrar_resumen(self):
        """Muestra una ventana emergente con las gráficas de resumen."""
        ventana_resumen = Toplevel(self.ventana)
        ventana_resumen.title("Resumen de Emociones")
        ventana_resumen.geometry("800x600")

        figura, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        figura.suptitle("Resumen de Emociones")

        ax1.pie(self.resultados_clientes.values(), labels=self.resultados_clientes.keys(), autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        ax1.set_title("Cliente")

        ax2.pie(self.resultados_general.values(), labels=self.resultados_general.keys(), autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)
        ax2.set_title("General")

        canvas = FigureCanvasTkAgg(figura, master=ventana_resumen)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def actualizar_grafico(self):
        """Actualiza las gráficas en la interfaz."""
        self.ax_clientes.clear()
        self.ax_clientes.set_title("Cliente")
        self.ax_clientes.axis("equal")

        self.ax_general.clear()
        self.ax_general.set_title("General")
        self.ax_general.axis("equal")

        if self.resultados_clientes:
            self.ax_clientes.pie(self.resultados_clientes.values(), labels=self.resultados_clientes.keys(), autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)

        if self.resultados_general:
            self.ax_general.pie(self.resultados_general.values(), labels=self.resultados_general.keys(), autopct='%1.1f%%', colors=plt.cm.Pastel1.colors)

        self.canvas_grafico.draw()

    def seleccionar_video(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de video", "*.mp4;*.avi")])
        if archivo:
            Thread(target=self.analizar_video, args=(archivo,)).start()

    def usar_camara(self):
        Thread(target=self.analizar_video, args=(1,)).start()

    def iniciar(self):
        self.ventana.mainloop()


# Ejecutar la aplicación
analizador = Analizador()
analizador.iniciar()

