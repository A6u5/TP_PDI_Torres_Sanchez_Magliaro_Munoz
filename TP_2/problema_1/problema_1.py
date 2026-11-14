# TP2 - Problema 1: detección y clasificación de monedas y dados
#
# Etapas:
#   A) Segmentación: Canny + cierre morfológico + contornos + envolvente convexa -> máscara.
#   B) Monedas vs dados (forma) y clasificación de monedas por color y tamaño + conteo.
#   C) Lectura de la cara superior de cada dado (conteo de puntos) + conteo.

# Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------------------------------------

# Carpeta donde está este script (problema_1/)
BASE_DIR = Path(__file__).resolve().parent

ESCALA = 0.65            # factor de reducción de la imagen (se aplica a color y a grises)
AREA_MIN = 500           # área mínima (px, ya reducida la imagen) para no tomar ruido como objeto
CIRC_MIN_MONEDA = 0.80   # área_objeto / área_del_círculo_mínimo_que_lo_envuelve (círculo ~1, cuadrado ~0.64)
GAP_TAMANIO = 0.08       # salto relativo de radio a partir del cual se considera otro tipo de moneda


# ---------------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------------------

def mostrar(imagen, titulo, es_color=False):
    """Muestra una imagen sola en una figura (para informar cada etapa)."""
    plt.figure(figsize=(10, 6))
    if es_color:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(imagen, cmap="gray")
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def mascara_de(forma, hull):
    """Máscara binaria (0/255) con la envolvente convexa rellena."""
    mascara = np.zeros(forma, dtype=np.uint8)
    cv2.drawContours(mascara, [hull], -1, 255, -1)
    return mascara


def agrupar_por_tamanio(radios, gap=GAP_TAMANIO):
    """Agrupa radios en tipos: ordena los radios y corta donde hay un salto relativo grande.

    Devuelve una lista de enteros (el tipo de cada radio, 0 = el más chico).
    """
    orden = np.argsort(radios)
    tipos = np.zeros(len(radios), dtype=int)
    tipo_actual = 0

    for k in range(1, len(orden)):
        anterior = radios[orden[k - 1]]
        actual = radios[orden[k]]

        if (actual - anterior) / anterior > gap:
            tipo_actual += 1

        tipos[orden[k]] = tipo_actual

    return tipos


# ---------------------------------------------------------------------------------------
# LECTURA DE LA IMAGEN
# ---------------------------------------------------------------------------------------

# Se lee a color (hace falta para distinguir monedas doradas / plateadas / bimetálicas)
# y se obtiene la versión en grises. Ambas se reducen igual para que coincidan pixel a pixel.
img_color = cv2.imread(str(BASE_DIR / "monedas.jpg"), cv2.IMREAD_COLOR)

if img_color is None:
    raise FileNotFoundError("No se encontró monedas.jpg junto a este script.")

img_color = cv2.resize(img_color, None, fx=ESCALA, fy=ESCALA, interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

mostrar(img, "Imagen original (grises, reducida)")

# Un umbral global no sirve porque el fondo no es uniforme: por eso se trabaja con bordes.


# ---------------------------------------------------------------------------------------
# ITEM A) SEGMENTACIÓN
# ---------------------------------------------------------------------------------------

# Blur gaussiano: reduce ruido y textura fina para que Canny no detecte bordes falsos.
img_suave = cv2.GaussianBlur(img, (5, 5), 1)

# Canny: bordes con histéresis (fuertes > 100; débiles entre 45 y 100 si se conectan a uno fuerte).
img_canny = cv2.Canny(img_suave, 45, 100)

# Cierre morfológico con elipse de 9x9: une los bordes cortados y deja contornos cerrados.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)

# Contornos externos (ignora lo de adentro: puntos de los dados, relieves de las monedas).
contornos, _ = cv2.findContours(img_cierre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Cantidad total de contornos detectados:", len(contornos))

# Descarto los contornos chicos (ruido). Me quedo con la envolvente convexa de cada objeto,
# que rellena entrantes y agujeros y deja una forma sólida.
objetos = []

for cnt in contornos:
    if cv2.contourArea(cnt) > AREA_MIN:
        objetos.append(cv2.convexHull(cnt))

print("Objetos detectados (monedas + dados):", len(objetos))

# Máscara final: todos los objetos en blanco sobre negro.
mascara_objetos = np.zeros_like(img)
cv2.drawContours(mascara_objetos, objetos, -1, 255, -1)

# Contornos sobre la imagen original, para comparar.
img_contornos = img_color.copy()
cv2.drawContours(img_contornos, objetos, -1, (255, 0, 0), 2)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_cierre, cmap="gray")
plt.title("Bordes + cierre morfológico")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB))
plt.title("Objetos detectados")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(mascara_objetos, cmap="gray")
plt.title("Máscara")
plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------------------------------------
# ITEM B) MONEDAS vs DADOS Y CLASIFICACIÓN DE MONEDAS
# ---------------------------------------------------------------------------------------

monedas = []
dados = []

for hull in objetos:
    area = cv2.contourArea(hull)
    (_, _), radio_min = cv2.minEnclosingCircle(hull)

    # Qué fracción del círculo mínimo que lo envuelve ocupa el objeto:
    # círculo ~ 1, cuadrado ~ 0.64 (aunque esté girado). Separa mejor que area/perimetro^2.
    llenado = area / (np.pi * radio_min ** 2)

    if llenado >= CIRC_MIN_MONEDA:
        monedas.append(hull)
    else:
        dados.append(hull)

print("\nMonedas detectadas:", len(monedas))
print("Dados detectados:", len(dados))

img_clasificada = img_color.copy()
cv2.drawContours(img_clasificada, monedas, -1, (0, 255, 0), 3)
cv2.drawContours(img_clasificada, dados, -1, (0, 0, 255), 3)
mostrar(img_clasificada, "Monedas (verde) - Dados (rojo)", es_color=True)


# --- Clasificación de las monedas por color y por tamaño ---

# Se mide el color en el espacio Lab: el canal b* va de azul a amarillo, así que las
# monedas doradas tienen b* alto y las plateadas lo tienen bajo (cerca del gris del fondo).
# En OpenCV (8 bits) el canal b* tiene un offset de 128.
img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
canal_b = img_lab[:, :, 2].astype(np.float32)

filas, columnas = np.ogrid[:img.shape[0], :img.shape[1]]

radios = []        # radio equivalente de cada moneda (sale del área)
b_centro = []      # b* medio del centro de la moneda
b_anillo = []      # b* medio del anillo exterior de la moneda

for hull in monedas:
    area = cv2.contourArea(hull)
    radios.append(np.sqrt(area / np.pi))

    (cx, cy), r = cv2.minEnclosingCircle(hull)
    distancia = np.sqrt((columnas - cx) ** 2 + (filas - cy) ** 2)

    # Centro (r < 45%) y anillo (entre 70% y 92% del radio). Así las bimetálicas
    # se distinguen: centro y borde de distinto color.
    centro = distancia < 0.45 * r
    anillo = (distancia > 0.70 * r) & (distancia < 0.92 * r)

    b_centro.append(canal_b[centro].mean())
    b_anillo.append(canal_b[anillo].mean())

radios = np.array(radios)
b_centro = np.array(b_centro)
b_anillo = np.array(b_anillo)

# Umbral automático (Otsu) entre "tira a amarillo" y "tira a gris" sobre el b* del anillo.
valores = np.clip(b_anillo, 0, 255).astype(np.uint8).reshape(-1, 1)
corte_otsu, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Otsu devuelve el último valor de la clase baja; para no quedar justo en el borde se usa
# el punto medio entre el promedio de la clase "gris" y el de la clase "amarilla".
grupo_gris = b_anillo[b_anillo <= corte_otsu]
grupo_amarillo = b_anillo[b_anillo > corte_otsu]
umbral_b = (grupo_gris.mean() + grupo_amarillo.mean()) / 2

print(f"\nUmbral de color (b*) elegido automáticamente: {umbral_b:.1f}")

# Categoría de color de cada moneda:
#   plateada: anillo poco amarillo
#   dorada:   anillo y centro amarillos
#   bimetálica: anillo amarillo pero centro no (o al revés: distinto color entre centro y anillo)
categorias = []

for bc, ba in zip(b_centro, b_anillo):
    anillo_dorado = ba > umbral_b
    centro_dorado = bc > umbral_b

    if anillo_dorado and centro_dorado:
        categorias.append("dorada")
    elif (not anillo_dorado) and (not centro_dorado):
        categorias.append("plateada")
    else:
        categorias.append("bimetalica")

categorias = np.array(categorias)

# Dentro de cada categoría de color, se separa por tamaño (radio).
tipos = np.zeros(len(monedas), dtype=int)

for cat in np.unique(categorias):
    indices = np.where(categorias == cat)[0]
    tipos[indices] = agrupar_por_tamanio(radios[indices])

# Etiqueta final de cada moneda, p. ej. "dorada-1" (la más chica de las doradas), "dorada-2", ...
etiquetas = [f"{categorias[i]}-{tipos[i] + 1}" for i in range(len(monedas))]

print("\nClasificación de las monedas (tipo: cantidad, radio medio en px):")
resumen = {}

for i, et in enumerate(etiquetas):
    resumen.setdefault(et, []).append(radios[i])

for et in sorted(resumen):
    print(f"  {et}: {len(resumen[et])} moneda(s), radio medio {np.mean(resumen[et]):.1f} px")

print("Total de monedas:", len(monedas))

# Para poner el valor real de cada tipo (10 centavos, 50 centavos, 1 peso, etc.),
# mirar la imagen y completar este diccionario con las etiquetas que salgan arriba:
#
# VALORES = {"plateada-1": "10 centavos", "dorada-1": "50 centavos", "bimetalica-1": "1 peso"}
# etiquetas = [VALORES.get(et, et) for et in etiquetas]

# Imagen con cada moneda rotulada.
colores_cat = {"dorada": (0, 215, 255), "plateada": (200, 200, 200), "bimetalica": (255, 0, 255)}
img_monedas = img_color.copy()

for hull, et, cat in zip(monedas, etiquetas, categorias):
    cv2.drawContours(img_monedas, [hull], -1, colores_cat[cat], 3)

    x, y, w, h = cv2.boundingRect(hull)
    cv2.putText(img_monedas, et, (x - 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_monedas, et, (x - 10, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

mostrar(img_monedas, "Tipos de monedas", es_color=True)


# ---------------------------------------------------------------------------------------
# ITEM C) LECTURA DE LOS DADOS
# ---------------------------------------------------------------------------------------

def contar_puntos(img_gris, hull):
    """Cuenta los puntos de un dado. Devuelve (cantidad, máscara de puntos)."""

    mascara = mascara_de(img_gris.shape, hull)
    area_dado = cv2.contourArea(hull)

    # Se erosiona la máscara para dejar afuera el borde del dado y su sombra.
    lado = int(np.sqrt(area_dado))
    k = max(3, int(0.06 * lado) | 1)   # tamaño impar
    mascara_interior = cv2.erode(mascara, np.ones((k, k), np.uint8))

    valores_dado = img_gris[mascara_interior > 0].reshape(-1, 1)

    # Otsu solo con los píxeles del dado: separa el blanco del dado de los puntos negros.
    umbral, _ = cv2.threshold(valores_dado, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    puntos = ((img_gris < umbral) & (mascara_interior > 0)).astype(np.uint8) * 255

    # Apertura para sacar píxeles sueltos.
    puntos = cv2.morphologyEx(puntos, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, etiquetas_cc, stats, _ = cv2.connectedComponentsWithStats(puntos, connectivity=8)

    cantidad = 0
    puntos_validos = np.zeros_like(puntos)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]

        # Un punto del dado es chico respecto del dado y aproximadamente redondo.
        tamanio_ok = 0.005 * area_dado < area < 0.10 * area_dado
        forma_ok = 0.5 < ancho / alto < 2.0

        if tamanio_ok and forma_ok:
            cantidad += 1
            puntos_validos[etiquetas_cc == i] = 255

    return cantidad, puntos_validos


img_dados = img_color.copy()
valores_dados = []

plt.figure(figsize=(4 * max(len(dados), 1), 4))

for n, hull in enumerate(dados):
    valor, puntos_validos = contar_puntos(img, hull)
    valores_dados.append(valor)

    print(f"Dado {n + 1}: cara superior = {valor}")

    cv2.drawContours(img_dados, [hull], -1, (0, 0, 255), 3)
    x, y, w, h = cv2.boundingRect(hull)
    cv2.putText(img_dados, str(valor), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img_dados, str(valor), (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255), 2, cv2.LINE_AA)

    plt.subplot(1, len(dados), n + 1)
    plt.imshow(puntos_validos[y:y + h, x:x + w], cmap="gray")
    plt.title(f"Dado {n + 1}: {valor}")
    plt.axis("off")

plt.tight_layout()
plt.show()

print("\nCantidad de dados:", len(dados))
print("Valores de las caras superiores:", valores_dados)

mostrar(img_dados, "Dados con el valor de su cara superior", es_color=True)