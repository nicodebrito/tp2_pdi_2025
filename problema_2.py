import matplotlib.pyplot as plt
import numpy as np
import cv2

def rellenar(img):
   img_flood_fill = img.copy().astype('uint8')
   h, w = img.shape[:2]
   mask = np.zeros((h+2, w+2), np.uint8)
   cv2.floodFill(img_flood_fill, mask, (0,0), 255)
   img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
   img_fh = img | img_flood_fill_inv
   return img_fh 


nro_patente = '10'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (5,5), 1.5) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    img_canny = cv2.Canny(blur, threshold1=50, threshold2=120) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5)) #15-3
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_dil = cv2.dilate(opening, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    img_rellena = rellenar(img_dil)
    #plt.imshow(img_rellena, cmap='gray'), plt.show()
    
    
    # --- Buscar contornos ---
    contours, hierarchy = cv2.findContours(img_rellena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img_rgb.copy()
    #for cnt in contours:
    #    area = cv2.contourArea(cnt)
    #    if area > 1000:
    #        x, y, w, h = cv2.boundingRect(cnt)
    #        a=cv2.rectangle(output, (x, y), (x+w, y+h), (0,0,255), 2)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        area_contorno = cv2.contourArea(cnt)
        area_bb = w * h
        aspect = w / h

        if area_bb == 0:
            continue
        
        if area_contorno < 1000:
            continue     

        if (area_contorno/area_bb) > 0.7:
            a=cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        if not (1.2 < aspect < 2.8):   # ajustable según objeto
            continue
        
        #if area > 1000:
        #    cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    plt.imshow(output), plt.show()

    """
    #Componentes
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_rellena) 
    output = img_rgb.copy()
    mask_final = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(1, n):  # 0 es fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = w / h

        if (area < 1400) or (area > 2300):   # ajustable
            continue

        if not (1.2 < aspect < 2.8):   # ajustable según objeto
            continue

        mask_final[labels == i] = 1
        print('Item:',str(i), ' Stats:', stats[i])
        a=cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2)    
        b=cv2.putText(output, str(i), (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    """
    plt.imshow(output), plt.show()
    
    
    output2 = img_rgb.copy()
    output2[mask_final == 1] = (0,255,0)
    plt.imshow(output2), plt.show()





ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_dil = cv2.dilate(img_canny, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
img_OPEN = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, elemento_cierre)
plt.imshow(img_OPEN, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
img_cierre = cv2.morphologyEx(img_OPEN, cv2.MORPH_CLOSE, elemento_cierre)
plt.imshow(img_cierre, cmap='gray'), plt.show()

img_rellena = rellenar(img_OPEN)
plt.imshow(img_rellena, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
img_OPEN = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, elemento_cierre)
plt.imshow(img_OPEN, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
img_cierre = cv2.morphologyEx(img_OPEN, cv2.MORPH_CLOSE, elemento_cierre)
plt.imshow(img_cierre, cmap='gray'), plt.show()

img_rellena = rellenar(img_OPEN)
plt.imshow(img_rellena, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10))
img_OPEN = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, elemento_cierre)
plt.imshow(img_OPEN, cmap='gray'), plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_OPEN) 
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)


ratio = stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH]
filtro = (ratio > 0.3) & (ratio < 0.6)

labels_filtrado = np.argwhere(filtro).flatten().tolist()

mask = np.isin(labels, labels_filtrado).astype(np.uint8)
plt.imshow(mask, cmap='gray'), plt.show()

print([stats[l] for l in labels_filtrado])
 # Nos quedamos con el último índice donde está la patente
idx_patente = labels_filtrado[4]
stats_patente = stats[idx_patente]
# Creación de la máscara para segmentar la patente
#sub_imagen = imutils.obtener_sub_imagen(img, stats_patente)

coor_h = stats_patente[0] 
coor_v = stats_patente[1]
ancho  = stats_patente[2]   
largo  = stats_patente[3]
sub_imagen = img[coor_v:coor_v + largo, coor_h: coor_h + ancho]

mascara = (labels == idx_patente).astype('uint8') * 255

pataente_segmentada = np.bitwise_and(img, mascara)

plt.imshow(pataente_segmentada, cmap='gray'), plt.show()

np.unique(mascara)

plt.imshow(mascara, cmap='gray'), plt.show()










img_umbralada = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img_umbralada, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
img_cierre = cv2.morphologyEx(img_umbralada, cv2.MORPH_OPEN, elemento_cierre)
plt.imshow(img_cierre, cmap='gray'), plt.show()


ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_dil = cv2.dilate(img_cierre, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_cierre)
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)

filtro = (0.3<(stats[:, cv2.CC_STAT_HEIGHT] / stats[:, cv2.CC_STAT_WIDTH])<0.6)

labels_filtrado = np.argwhere(filtro).flatten().tolist()

 # Nos quedamos con el último índice donde está la patente
idx_patente = labels_filtrado[-1]
stats_patente = stats[17]
# Creación de la máscara para segmentar la patente
#sub_imagen = imutils.obtener_sub_imagen(img, stats_patente)

coor_h = stats_patente[0] 
coor_v = stats_patente[1]
ancho  = stats_patente[2]   
largo  = stats_patente[3]
sub_imagen = img[coor_v:coor_v + largo, coor_h: coor_h + ancho]

mascara = (labels == idx_patente).astype('uint8') * 255

pataente_segmentada = np.bitwise_and(img, mascara)

plt.imshow(pataente_segmentada, cmap='gray'), plt.show()

np.unique(mascara)

plt.imshow(mascara, cmap='gray'), plt.show()


elemento_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
img_OPEN= cv2.morphologyEx(img_umbralada, cv2.MORPH_OPEN, elemento_cierre)
plt.imshow(img_OPEN, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 3))
img_cierre = cv2.morphologyEx(img_OPEN, cv2.MORPH_CLOSE, elemento_cierre)
plt.imshow(img_cierre, cmap='gray'), plt.show()


# Blur y detección de bordes
blur = cv2.GaussianBlur(img, (5,5), 0)
plt.imshow(blur, cmap='gray'), plt.show()

img_eq = cv2.equalizeHist(blur)
plt.imshow(img_eq, cmap='gray'), plt.show()

img_canny = cv2.Canny(blur, 10, 150)
plt.imshow(img_canny, cmap='gray'), plt.show()

elemento_cierre = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 11))
img_cierre = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, elemento_cierre)
plt.imshow(img_cierre, cmap='gray'), plt.show()

# Componente conectados, filtramos por área
n, labels, stats, _  = cv2.connectedComponentsWithStats(img_cierre)
AREA_MIN = 400
filtro = (
    (stats[:, cv2.CC_STAT_AREA] >= AREA_MIN) & 
    (stats[:, cv2.CC_STAT_HEIGHT] < stats[:, cv2.CC_STAT_WIDTH]))
labels_filtrado = np.argwhere(filtro).flatten().tolist()
# Nos quedamos con el último índice donde está la patente
idx_patente = labels_filtrado[-1]
stats_patente = stats[idx_patente]
# Creación de la máscara para segmentar la patente
sub_imagen = imutils.obtener_sub_imagen(img, stats_patente)
mascara = (labels == idx_patente).astype('uint8') * 255
K_ANCHO =  15 # floor(sub_imagen.shape[0]  * 0.20)  # 15
K_LARGO =  3  # floor(sub_imagen.shape[1]  * 0.15)  # 3
elemento_dil = cv2.getStructuringElement(
                    cv2.MORPH_RECT, 
                    (K_ANCHO, K_LARGO))
img_dil = cv2.dilate(mascara, elemento_dil)

plt.imshow(mascara, cmap='gray'), plt.show()

pataente_segmentada = np.bitwise_and(img, img_dil)

plt.imshow(pataente_segmentada, cmap='gray'), plt.show()

if plot:
    plt.figure()
    plt.imshow(pataente_segmentada, cmap='gray')
    plt.title(basename(path))
    plt.show()




plt.imshow(img, cmap='gray'), plt.show()


img_umbralada = cv2.threshold(img, 121, 255, cv2.THRESH_BINARY)[1]
plt.imshow(img_umbralada, cmap='gray'), plt.show()

# Componentes conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbralada)    
# Filtramos elementos con áreas muy pequeñas y muy chicas
# Esta copia del labels (que la vamos modificando) y la siguiente fueron hechas para no sobrescribir
# variables y poder ir controlando el flujo del código a medida que aparecían errores
labels_copia = labels.copy()
# Recorremos cada componente conectado sin incluir el fondo
for i in range(num_labels):
    # Los que tengan un area superior e inferior a determinados umbrales
    if stats[i, -1] < 26 or stats[i, -1] > 98:
        
        # Los eliminamos
        labels_copia[labels_copia == i] = 0

plt.imshow(labels_copia, cmap='gray'), plt.show()