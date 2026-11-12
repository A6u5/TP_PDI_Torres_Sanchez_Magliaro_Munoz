# Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



# Carpeta donde está este script (problema_1/)
BASE_DIR = Path(__file__).resolve().parent

# Lectura de la imagen de las monedas y los dados en escala de grises
img = cv2.imread(str(BASE_DIR / "monedas.jpg"), cv2.IMREAD_GRAYSCALE)



# Si se intenta un umbralado global como se hizo en otras ocaciones no funciona ya que el resultado confunde mas que ayudar

"""
img_th = img < 128
plt.imshow(img_th, cmap="gray")
plt.title("Imagen Umbralada")
plt.axis("off")
plt.tight_layout()
plt.show()

cv2.IMREAD_GRAYSCALE

#HISTOGRAMA DE INTENSIDADES

plt.figure(figsize=(8,4))
plt.hist(img_gray.ravel(), bins=256, range=[0,256])
plt.title("Histograma de intensidades")
plt.xlabel("Intensidad")
plt.ylabel("Cantidad de píxeles")
plt.show()
"""

#### ITEM A) ####

# Redimensiono la imagen a un 65% para que se vea mejor en el plot
img = cv2.resize(img, None, fx=0.65, fy=0.65)

# Blur gaussiano (kernel 5x5, sigma=1): suaviza la imagen a través de reducir 
# el ruido y la textura fina para que Canny no detecte bordes falsos
img_suave = cv2.GaussianBlur(img, (5, 5), 1)
# Canny: detecta bordes. Umbral bajo=45 y alto=100 (histéresis: bordes fuertes > 100, 
# débiles entre 45 y 100 si se conectan con fuertes)
img_canny = cv2.Canny(img_suave, 45, 100)

# Elemento estructurante elíptico de 9x9 para la operación morfológica
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (9, 9))

# Cierre morfológico (dilatación + erosión): une los bordes cortados de Canny y cierra huecos pequeños,
# de modo que cada objeto quede con un contorno cerrado
img_cierre = cv2.morphologyEx(
    img_canny, cv2.MORPH_CLOSE, kernel)

# Busco los contornos en la imagen binaria.
# RETR_EXTERNAL: solo contornos externos (ignora los internos, como los puntos de los dados)
# CHAIN_APPROX_SIMPLE: comprime los segmentos rectos guardando solo sus extremos
contornos, jerarquia = cv2.findContours(
    img_cierre,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

# Cuento cuanta cantidad de contornos detecto
print("Cantidad total de contornos detectados:", len(contornos))

# Filtro contornos por área, los mas pequeños son ruido, los mas grandes objetos (dados monedas)
area_min = 500

# Lista donde guardo solo los contornos que superan el área mínima
contornos_validos = []

for cnt in contornos:
    # Área encerrada por el contorno (en píxeles)
    area = cv2.contourArea(cnt)

    # Me quedo solo con los contornos grandes (descarto ruido)
    if area > area_min:
        contornos_validos.append(cnt)

# Cantidad de objetos (monedas + dados) que quedaron tras el filtro
print("Objetos detectados:", len(contornos_validos))


#Hago la mascara.
# Imagen negra del mismo tamaño que img, donde se pintarán los objetos en blanco
mascara_objetos = np.zeros_like(img)

for cnt in contornos_validos:

    # Envolvente convexa: rellena entrantes/huecos del contorno y da una forma sólida y convexa
    hull = cv2.convexHull(cnt)

    # Dibujo la envolvente rellena (thickness=-1) en blanco (255) sobre la máscara
    cv2.drawContours(
        mascara_objetos,   # imagen donde se dibuja
        [hull],            # lista de contornos a dibujar
        -1,                # dibujar todos los contornos de la lista
        255,               # color: blanco
        -1                 # grosor -1 = relleno
    )


# Dibujamos los contornos sobre la imagen original para verlos en comparacion (Esto es opcional pero queda lindo)
# Paso la imagen a 3 canales (BGR) para poder dibujar en color
img_contornos = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Dibujo los contornos válidos en azul (BGR: 255,0,0) con grosor 2
cv2.drawContours(
    img_contornos,
    contornos_validos,
    -1,
    (255, 0, 0),
    2
)


### TODOS LOS PLOTS ####

# Figura con 3 paneles: cierre morfológico, contornos sobre la original y máscara
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_cierre, cmap="gray")
plt.title("Cierre morfológico")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title("Contornos válidos")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mascara_objetos, cmap="gray")
plt.title("Mascara")
plt.axis("off")

# Ajusta los márgenes y muestra la figura
plt.tight_layout()
plt.show()


####### ITEM B ##########

# Listas donde se guardan los contornos clasificados
monedas = []
dados = []

for cnt in contornos_validos:

    # Envolvente convexa del objeto
    hull = cv2.convexHull(cnt)

    # Área y perímetro de la envolvente (True = contorno cerrado)
    area = cv2.contourArea(hull)
    perimetro = cv2.arcLength(hull, True)

    # Factor de forma = área / perímetro². Es invariante a la escala:
    # círculo ≈ 1/(4π) ≈ 0.0796; cuadrado = 1/16 = 0.0625
    fp = area / (perimetro ** 2)

    # Cerca del círculo -> moneda; si no -> dado (umbral 0.075 entre ambos valores)
    if fp > 0.075:
        monedas.append(hull)
    else:
        dados.append(hull)

# Cantidad de cada tipo de objeto
print("Monedas detectadas:", len(monedas))
print("Dados detectados:", len(dados))

# Imagen en color para dibujar la clasificación
img_clasificada = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Monedas en verde (0,255,0) y dados en rojo (0,0,255), grosor 3
cv2.drawContours(img_clasificada, monedas, -1, (0, 255, 0), 3)
cv2.drawContours(img_clasificada, dados, -1, (0, 0, 255), 3)

# Muestro el resultado final (se convierte BGR -> RGB porque matplotlib espera RGB)
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img_clasificada, cv2.COLOR_BGR2RGB))
plt.title("Monedas (verde) - Dados (rojo)")
plt.axis("off")
plt.show()