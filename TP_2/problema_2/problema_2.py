# TP2 - Problema 2: detección de patentes y segmentación de sus caracteres
#
# Datos del enunciado que se usan:
#   - Relación de aspecto de cada caracter: alto/ancho entre 1.5 y 3.0.
#   - Los caracteres forman grupos de 3.
#   - La patente son dos grupos de 3 caracteres (6 en total, formato "ABC 123").
#
# Las patentes son de fondo negro y caracteres blancos (formato "ABC 123").
#
# Etapas (por cada imagen):
#   A) Detección de la patente: se binariza, se buscan componentes conectadas con forma de
#      caracter y se busca la secuencia de 6 que forme dos grupos de 3 alineados.
#      La patente es el rectángulo que envuelve esos 6 caracteres (con un margen).
#   B) Segmentación de los caracteres: cada uno de los 6 se recorta por separado.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------------------
# PARÁMETROS
# ---------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CARPETA_IMAGENES = BASE_DIR / "patentes"   # si no existe, se busca junto al script

# Se prueba primero con criterios ESTRICTOS (los del enunciado). Si en una imagen no aparece
# ninguna patente, se vuelve a intentar con criterios RELAJADOS (por ejemplo, caracteres un poco
# deformados por la perspectiva o el desenfoque).
CRITERIOS = [
    {"nombre": "estricto",
     "aspecto": (1.5, 3.0),        # alto/ancho de un caracter (dato del enunciado)
     "relleno": (0.25, 0.90),      # fracción de su caja que ocupa un caracter
     "tol_alto": 0.25,             # los 6 caracteres tienen alto parecido (+-25 %)
     "tol_fila": 0.60,             # centros dentro de 60 % del alto (foto algo inclinada)
     "salto_fila": 0.35,           # salto vertical máximo entre caracteres vecinos (x alto)
     "ratio_grupos": 1.3,          # el hueco entre grupos debe ser > 1.3 x el mayor hueco interno
     "ancho_max": 9},              # ancho máximo de la patente (x alto de caracter)
    {"nombre": "relajado",
     "aspecto": (1.2, 3.8),
     "relleno": (0.20, 0.95),
     "tol_alto": 0.40,
     "tol_fila": 0.80,
     "salto_fila": 0.50,
     "ratio_grupos": 1.1,
     "ancho_max": 11},
]

ALTO_MIN_REL = 0.015                  # alto mínimo de un caracter, relativo al alto de la imagen
ALTO_MAX_REL = 0.20                   # alto máximo de un caracter, relativo al alto de la imagen
MARGEN_X = 0.45                       # margen horizontal de la patente alrededor de los caracteres (x alto)
MARGEN_Y = 0.35                       # margen vertical de la patente alrededor de los caracteres (x alto)

# Las patentes de este formato (ABC 123) tienen los caracteres BLANCOS sobre fondo NEGRO.
# Se prueban varios umbrales globales (según la iluminación de cada foto cambia cuál separa
# mejor) y también un umbral adaptativo (que se ajusta zona por zona: sombras, reflejos).
UMBRALES = list(range(70, 231, 10))

# Si en la imagen original no aparece la patente (caracteres muy chicos, de pocos píxeles de
# ancho, que se rompen al binarizar), se repite la búsqueda ampliando la imagen.
ESCALAS = [1, 2, 3]


# ---------------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------------------

def componentes_candidatas(binaria, alto_img, crit):
    """Componentes conectadas con forma de caracter (aspecto y tamaño razonables)."""

    num, _, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    candidatas = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]

        if w == 0:
            continue

        aspecto = h / w
        alto_ok = ALTO_MIN_REL * alto_img <= h <= ALTO_MAX_REL * alto_img
        aspecto_ok = crit["aspecto"][0] <= aspecto <= crit["aspecto"][1]

        # Un caracter ocupa una buena parte de su caja (descarta bordes y líneas finas).
        relleno_ok = crit["relleno"][0] <= area / (w * h) <= crit["relleno"][1]

        if alto_ok and aspecto_ok and relleno_ok:
            candidatas.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    return candidatas


def mejor_secuencia(candidatas, crit, es_valida=None):
    """Busca 6 caracteres alineados, de alto parecido, formando dos grupos de 3.

    Devuelve (lista de 6 componentes ordenadas de izquierda a derecha, puntaje) o (None, inf).
    El puntaje es menor cuanto más regular es la secuencia.
    """

    mejor, mejor_puntaje = None, np.inf

    for base in candidatas:
        alto = base["h"]
        cy_base = base["y"] + base["h"] / 2

        # Vecinos: mismo alto y misma fila que la componente base.
        fila = [c for c in candidatas
                if abs(c["h"] - alto) <= crit["tol_alto"] * alto
                and abs((c["y"] + c["h"] / 2) - cy_base) <= crit["tol_fila"] * alto]

        fila.sort(key=lambda c: c["x"])

        # Se prueba cada ventana de 6 componentes consecutivas de la fila.
        for k in range(len(fila) - 5):
            ventana = fila[k:k + 6]

            # Separación entre caracteres vecinos (borde derecho -> borde izquierdo siguiente).
            huecos = [ventana[i + 1]["x"] - (ventana[i]["x"] + ventana[i]["w"]) for i in range(5)]

            if min(huecos) < 0:
                continue

            # Alineación caracter a caracter (tolera una foto levemente inclinada).
            centros_y = [c["y"] + c["h"] / 2 for c in ventana]

            if max(abs(centros_y[i + 1] - centros_y[i]) for i in range(5)) > crit["salto_fila"] * alto:
                continue

            alto_medio = np.mean([c["h"] for c in ventana])

            # Dos grupos de 3: los huecos dentro de cada grupo (0, 1 y 3, 4) son chicos
            # y el hueco entre grupos (el 2) es claramente mayor.
            dentro = [huecos[0], huecos[1], huecos[3], huecos[4]]
            entre = huecos[2]

            if entre <= crit["ratio_grupos"] * max(dentro):
                continue

            # La patente entera no puede ser demasiado ancha respecto de sus caracteres.
            ancho_total = ventana[-1]["x"] + ventana[-1]["w"] - ventana[0]["x"]

            if ancho_total > crit["ancho_max"] * alto_medio:
                continue

            # Verificación extra (contraste claro/oscuro de la patente).
            if es_valida is not None and not es_valida(ventana):
                continue

            # Puntaje: variación de altos + irregularidad de los huecos dentro de los grupos.
            puntaje = (np.std([c["h"] for c in ventana]) / alto_medio
                       + np.std(dentro) / alto_medio)

            if puntaje < mejor_puntaje:
                mejor, mejor_puntaje = ventana, puntaje

    return mejor, mejor_puntaje


CONTRASTE_MIN = 70   # diferencia mínima de brillo entre los caracteres (claros) y el fondo de la patente (oscuro)


def contraste_patente(gris, binaria, ventana):
    """Brillo medio de los caracteres menos brillo medio del fondo que los rodea.

    En una patente real los caracteres son claros y el fondo es negro, así que la diferencia
    es grande. En falsos positivos (adoquines, rejillas, texturas) es mucho menor.
    """

    x0 = min(c["x"] for c in ventana)
    x1 = max(c["x"] + c["w"] for c in ventana)
    y0 = min(c["y"] for c in ventana)
    y1 = max(c["y"] + c["h"] for c in ventana)

    zona_gris = gris[y0:y1, x0:x1]
    zona_bin = binaria[y0:y1, x0:x1] > 0

    # Píxeles de caracter: solo los de las cajas de las 6 componentes.
    es_caracter = np.zeros_like(zona_bin)

    for c in ventana:
        es_caracter[c["y"] - y0:c["y"] - y0 + c["h"], c["x"] - x0:c["x"] - x0 + c["w"]] = True

    caracter = es_caracter & zona_bin

    # Fondo: lo que queda dentro de la zona sin los caracteres (ni su borde).
    cerca = cv2.dilate(zona_bin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    fondo = ~cerca

    if caracter.sum() == 0 or fondo.sum() == 0:
        return 0.0

    return float(zona_gris[caracter].mean() - zona_gris[fondo].mean())


def generar_binarias(gris):
    """Genera las imágenes binarias a probar (caracteres en blanco).

    Para cada una se agrega una versión con una pequeña erosión, que separa caracteres
    pegados entre sí o pegados al marco de la patente.
    """

    suave = cv2.GaussianBlur(gris, (3, 3), 0)

    # Realce de contraste local: ayuda en fotos con poco contraste o sombras.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    realzada = clahe.apply(suave)

    binarias = []

    for origen in (suave, realzada):
        for umbral in UMBRALES:
            binarias.append((f"umbral {umbral}", (origen > umbral).astype(np.uint8) * 255))

        for bloque in (15, 25, 41):
            adaptativa = cv2.adaptiveThreshold(origen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, bloque, -10)
            binarias.append((f"adaptativo {bloque}", adaptativa))

    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    con_erosion = [(nombre + " + erosión", cv2.erode(b, elemento)) for nombre, b in binarias]

    return binarias + con_erosion


def detectar_caracteres(gris):
    """Devuelve (6 caracteres, binaria usada, criterio usado) o (None, None, None)."""

    alto_img = gris.shape[0]
    binarias = generar_binarias(gris)

    for crit in CRITERIOS:
        mejor, mejor_puntaje, mejor_binaria = None, np.inf, None

        for nombre, binaria in binarias:
            candidatas = componentes_candidatas(binaria, alto_img, crit)
            secuencia, puntaje = mejor_secuencia(
                candidatas, crit,
                es_valida=lambda v, b=binaria: contraste_patente(gris, b, v) >= CONTRASTE_MIN)

            if secuencia is not None and puntaje < mejor_puntaje:
                mejor, mejor_puntaje, mejor_binaria = secuencia, puntaje, binaria

        if mejor is not None:
            return mejor, mejor_binaria, crit["nombre"]

    return None, None, None


def procesar_imagen(ruta, mostrar=True):
    """Aplica las dos etapas sobre una imagen. Devuelve (recorte_patente, lista_de_caracteres)."""

    color = cv2.imread(str(ruta), cv2.IMREAD_COLOR)

    if color is None:
        print(f"  No se pudo leer {ruta.name}")
        return None, []

    gris = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # -------- A) DETECCIÓN DE LA PATENTE --------
    caracteres = None

    for escala in ESCALAS:
        if escala == 1:
            color_e, gris_e = color, gris
        else:
            color_e = cv2.resize(color, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            gris_e = cv2.cvtColor(color_e, cv2.COLOR_BGR2GRAY)

        caracteres, binaria, criterio = detectar_caracteres(gris_e)

        if caracteres is not None:
            if escala > 1:
                print(f"  (los caracteres eran muy chicos: se amplió la imagen x{escala})")

            # De acá en adelante se trabaja con la imagen en la escala donde se detectó.
            color, gris = color_e, gris_e
            break

    if caracteres is None:
        print("  No se encontró una secuencia de 2 grupos de 3 caracteres.")

        # Diagnóstico: se muestran los objetos que parecen caracteres (criterio relajado) en la
        # binarización que más encontró, para ver qué falta (caracteres pegados, perdidos, etc.).
        mejor_n, mejores, nombre_mejor = 0, [], ""

        for nombre, bin_i in generar_binarias(gris):
            cands = componentes_candidatas(bin_i, gris.shape[0], CRITERIOS[-1])

            if len(cands) > mejor_n:
                mejor_n, mejores, nombre_mejor = len(cands), cands, nombre

        print(f"  Diagnóstico: máximo de {mejor_n} objetos con forma de caracter ({nombre_mejor}).")

        if mostrar:
            diag = color.copy()

            for c in mejores:
                cv2.rectangle(diag, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), (0, 0, 255), 1)

            plt.figure(figsize=(8, 5))
            plt.imshow(cv2.cvtColor(diag, cv2.COLOR_BGR2RGB))
            plt.title(f"{ruta.name}: patente NO detectada (rojo = objetos con forma de caracter)")
            plt.axis("off")
            plt.show()

        return None, []

    x0 = min(c["x"] for c in caracteres)
    x1 = max(c["x"] + c["w"] for c in caracteres)
    y0 = min(c["y"] for c in caracteres)
    y1 = max(c["y"] + c["h"] for c in caracteres)
    alto = np.mean([c["h"] for c in caracteres])

    # Se agrega un margen para incluir el fondo blanco de la patente.
    px0 = max(int(x0 - MARGEN_X * alto), 0)
    px1 = min(int(x1 + MARGEN_X * alto), gris.shape[1])
    py0 = max(int(y0 - MARGEN_Y * alto), 0)
    py1 = min(int(y1 + MARGEN_Y * alto), gris.shape[0])

    patente = color[py0:py1, px0:px1]
    print(f"  Patente detectada (criterio {criterio}) en x=[{px0}, {px1}], y=[{py0}, {py1}]")

    img_deteccion = color.copy()
    cv2.rectangle(img_deteccion, (px0, py0), (px1, py1), (0, 255, 0), 2)

    # -------- B) SEGMENTACIÓN DE CARACTERES --------
    # Se vuelve a binarizar solo el recorte de la patente, con Otsu (más fino que el
    # umbral global), y se recorta cada caracter por la caja detectada.
    gris_patente = gris[py0:py1, px0:px1]
    _, binaria_patente = cv2.threshold(gris_patente, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    recortes = []
    img_caracteres = patente.copy()

    for c in caracteres:
        cx0, cy0 = c["x"] - px0, c["y"] - py0
        recortes.append(binaria_patente[cy0:cy0 + c["h"], cx0:cx0 + c["w"]])
        cv2.rectangle(img_caracteres, (cx0, cy0), (cx0 + c["w"], cy0 + c["h"]), (0, 0, 255), 1)

    print(f"  Caracteres segmentados: {len(recortes)}")

    if mostrar:
        plt.figure(figsize=(14, 8))
        plt.suptitle(ruta.name)

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        plt.title("Imagen original")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(binaria, cmap="gray")
        plt.title("Binarización (mejor umbral)")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(img_deteccion, cv2.COLOR_BGR2RGB))
        plt.title("A) Patente detectada")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(patente, cv2.COLOR_BGR2RGB))
        plt.title("A) Patente segmentada")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(img_caracteres, cv2.COLOR_BGR2RGB))
        plt.title("B) Caracteres detectados")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(binaria_patente, cmap="gray")
        plt.title("B) Patente binarizada")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 2.5))

        for n, recorte in enumerate(recortes):
            plt.subplot(1, 6, n + 1)
            plt.imshow(recorte, cmap="gray")
            plt.title(f"Car. {n + 1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return patente, recortes


# ---------------------------------------------------------------------------------------
# PROGRAMA PRINCIPAL: se aplica el algoritmo de forma cíclica a las 12 imágenes
# ---------------------------------------------------------------------------------------

if __name__ == "__main__":

    carpeta = CARPETA_IMAGENES if CARPETA_IMAGENES.exists() else BASE_DIR
    detectadas = 0

    for n in range(1, 13):
        ruta = carpeta / f"img{n:02d}.png"

        print("=" * 50)
        print(f"IMAGEN {n:02d} ({ruta.name})")
        print("=" * 50)

        patente, recortes = procesar_imagen(ruta)

        if patente is not None:
            detectadas += 1

        print()

    print(f"Patentes detectadas: {detectadas} de 12")