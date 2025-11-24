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

    blur = cv2.GaussianBlur(img_gray, (7,7), 1.5) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    img_canny = cv2.Canny(blur, threshold1=50, threshold2=120) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1)) #15-3
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15)) #15-3
    closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    opening = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel)
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
        
        if area_contorno < 500:
            continue     

        if (area_contorno/area_bb) > 0.5:
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



#otro enfoque

def histograma(imagen, M, N):
    imagen_bordes = cv2.copyMakeBorder(imagen, M//2, M//2, N//2, N//2, cv2.BORDER_REPLICATE)        #Genera una imagen con bordes. Se utiliza bordes replicados.
    img_t = np.zeros_like(imagen)                                                                   #Imagen vacia con misma forma que la original
    filas, columnas = img_t.shape                                                                   #Dimensiones de la img

    for x in range(filas):                                                          
        for y in range(columnas):                                                    
            ventana = imagen_bordes[x:x + M, y: y + N]             #Genera ventana de la imagen con bordes
            ventana_equalize = cv2.equalizeHist(ventana)           #Aplica eq del histograma en la ventana. Como agregamos bordes no genera problemas en los bordes originales
            img_t[x, y] = ventana_equalize[M//2, N//2]             #Setea resultado en la imagen trasnformada en la posicion que corresponde.

    return img_t



img_h1 = histograma(img, 3, 3)
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    #nro_patente = '06'
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    img_t = histograma(blur,5,5)
    #plt.imshow(img_t, cmap='gray'), plt.show()

    mask_black = cv2.inRange(img_t, 200, 255)
    #plt.imshow(mask_black, cmap='gray'), plt.show()
 
    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    img_dil = cv2.dilate(mask_black, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    opening = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()



    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dil) 
    output = img_rgb.copy()

    mask_final = np.zeros(labels.shape, dtype=np.uint8)
    for i in range(1, n):  # 0 es fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        aspect = w / h
        if area>400:
            continue
        a=cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 2) 
        mask_final[labels == i] = 1

    plt.imshow(output), plt.show()

    img_canny = cv2.Canny(img_t, threshold1=200, threshold2=255) #50-120
    plt.imshow(img_canny, cmap='gray'), plt.show()
    
    sobelx = cv2.Sobel(mask_black, cv2.CV_8F, 1, 1,ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    plt.imshow(sobelx, cmap='gray'), plt.show()





    img_invertida = cv2.bitwise_not(mask_black)
    plt.imshow(img_invertida, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
    img_dil = cv2.dilate(img_invertida, ee_dil)
    plt.imshow(img_dil, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    opening = cv2.morphologyEx(img_invertida, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    elemento_estructural_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 2)) 
    img_erosionada = cv2.erode(mask_black, elemento_estructural_2, iterations=1) 
    plt.imshow(img_erosionada, cmap='gray'), plt.show()

    img_invertida = cv2.bitwise_not(img_erosionada)
    plt.imshow(img_invertida, cmap='gray'), plt.show()






    img_t = histograma(img_gray,25,25)
    plt.imshow(img_t, cmap='gray'), plt.show()

    mask_white = cv2.inRange(img_t, 120, 255)
    plt.imshow(mask_white, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    opening = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) #15-3
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='gray'), plt.show() 

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
    
    plt.imshow(output), plt.show()


    contours, hierarchy = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        a = cv2.drawContours(output, [cnt], -1, (255, 0, 0), 2)
        #if area_bb == 0:
        #    continue
        #
        #if area_contorno < 500:
        #    continue     

        #if (area_contorno/area_bb) > 0.5:
        #    a=cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        #if not (1.2 < aspect < 2.8):   # ajustable según objeto
        #    continue
        
        #if area > 1:
        #    cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    plt.imshow(output), plt.show()

    blur = cv2.GaussianBlur(img_t, (5,5), 0) #5,5  1.5
    plt.imshow(blur, cmap='gray'), plt.show()

    mask_white = cv2.inRange(img_t, 140, 255)
    plt.imshow(mask_white, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5)) #15-3
    closing = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
    opening = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    img_invertida = cv2.bitwise_not(opening)
    plt.imshow(img_invertida, cmap='gray'), plt.show()

    img_rellena = rellenar(img_invertida)
    plt.imshow(img_rellena, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15)) #15-3
    closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    opening = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_dil = cv2.dilate(opening, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    img_rellena = rellenar(img_dil)
    #plt.imshow(img_rellena, cmap='gray'), plt.show()



    img_umbralada = cv2.threshold(img_gray, 70, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(img_umbralada, cmap='gray'), plt.show()







nro_patente = '10'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #plt.imshow(img_hsv, cmap='gray'), plt.show()

    
    #v = img_hsv[:,:,2]
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #v_eq = clahe.apply(v)
    #plt.imshow(v_eq, cmap='gray'), plt.show()

    #blur = cv2.GaussianBlur(v_eq, (5,5), 0) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()


    img_h1 = histograma(img_gray, 9, 9)

    blur = cv2.bilateralFilter(img_h1, d=15, sigmaColor=200, sigmaSpace=200)
    #plt.imshow(blur, cmap='gray'), plt.show()

    #blur = cv2.GaussianBlur(img_gray, (7,7), 1.5) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()


    edges = cv2.Canny(blur, 50, 100)
    #plt.imshow(edges, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    #edges_dil = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    #edges_dil = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)


    #plt.imshow(closed, cmap='gray'), plt.show()


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,5))
    opening = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,2))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()



    # --- Buscar contornos ---
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        ratio = w / float(h)
        if 3 < ratio < 6 :
            a=cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        if area_bb == 0:
            continue
        
        if area_contorno < 500:
            continue     

        if (area_contorno/area_bb) > 0.5:
            a=cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        if not (1.2 < aspect < 2.8):   # ajustable según objeto
            continue
        
        #if area > 1000:
        #    cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    plt.imshow(output), plt.show()







    img_canny = cv2.Canny(blur, threshold1=50, threshold2=120) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1)) #15-3
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15)) #15-3
    closing2 = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    opening = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel)
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
        
        if area_contorno < 500:
            continue     

        if (area_contorno/area_bb) > 0.5:
            a=cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

        if not (1.2 < aspect < 2.8):   # ajustable según objeto
            continue
        
        #if area > 1000:
        #    cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)

    plt.imshow(output), plt.show()

    




nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #plt.imshow(img_rgb, cmap='gray'), plt.show()

    #img_blur = cv2.bilateralFilter(img, 9, 75, 75)
    #gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray, cmap='gray'), plt.show()

    #blur = cv2.GaussianBlur(img_gray, (5,5), 1) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    #median = cv2.medianBlur(blur,7)
    #plt.imshow(median, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (7,7), 0) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(0.75*sobelx + 0.75*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    hist = histograma(sobel,71,71)
    plt.imshow(hist, cmap='gray'), plt.show()


    th = 150
    th_out, img_th = cv2.threshold(hist, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
    #plt.imshow(img_th, cmap='gray'), plt.show()



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,4))
    closed1 = cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closed1, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    closed2 = cv2.morphologyEx(closed1, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closed2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()


    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    img_dil = cv2.dilate(opening, ee_dil)
    plt.imshow(img_dil, cmap='gray'), plt.show()




    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closed, cmap='gray'), plt.show()





    th = cv2.adaptiveThreshold(
        closed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2)
    #plt.imshow(th, cmap='gray'), plt.show()


    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_small)
    plt.imshow(clean, cmap='gray'), plt.show()








# ----------------------------------------------------------------------------------------------
# --- Filtro Sobel - Combinacion - Ejemplo sobre imagen real -----------------------------------
# ----------------------------------------------------------------------------------------------
# --- Cargo Imagen --------------------------------------------------------------
img = cv2.imread('cameraman.tif',cv2.IMREAD_GRAYSCALE)              
plt.figure(), plt.imshow(img, cmap='gray'), plt.show(block=False)

# --- Sobel ---------------------------------------------------------------------
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  
sobel_xy = sobel_x + sobel_y
sobel = cv2.convertScaleAbs(sobel_xy)

sobel_x2 = cv2.convertScaleAbs(sobel_x)
sobel_y2 = cv2.convertScaleAbs(sobel_y)
sobel_2 = cv2.addWeighted(sobel_x2, 0.5, sobel_y2, 0.5, 0)
sobel_3 = cv2.addWeighted(sobel_x2, 1, sobel_y2, 1, 0)

plt.figure()
ax1 = plt.subplot(221)
plt.imshow(img, cmap='gray'), plt.title('Imagen'), plt.colorbar()
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(sobel, cmap='gray'), plt.title('float - "+" - convertScaleAbs'), plt.colorbar()
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(sobel_2, cmap='gray'), plt.title('convertScaleAbs - addWeighted(0.5)'), plt.colorbar()
plt.subplot(224,sharex=ax1,sharey=ax1), plt.imshow(sobel_3, cmap='gray'), plt.title('convertScaleAbs - addWeighted(1.0)'), plt.colorbar()
plt.suptitle("Sobel")
plt.show(block=False)





nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #plt.imshow(img_rgb, cmap='gray'), plt.show()

    #img_blur = cv2.bilateralFilter(img, 9, 75, 75)
    #gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray, cmap='gray'), plt.show()

    #blur = cv2.GaussianBlur(img_gray, (5,5), 1) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    #median = cv2.medianBlur(blur,7)
    #plt.imshow(median, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobelx + sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    img_canny = cv2.Canny(sobel, threshold1=170, threshold2=255) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    img_dil = cv2.erode(img_canny, ee_dil)
    plt.imshow(img_dil, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    img_dil = cv2.dilate(img_canny, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    opening = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))
    closed2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closed2, cmap='gray'), plt.show()




def rellenar(img):
   img_flood_fill = img.copy().astype('uint8')
   h, w = img.shape[:2]
   mask = np.zeros((h+2, w+2), np.uint8)
   cv2.floodFill(img_flood_fill, mask, (0,0), 255)
   img_flood_fill_inv = cv2.bitwise_not(img_flood_fill)
   img_fh = img | img_flood_fill_inv
   return img_fh 


nro_patente = '01'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.imshow(img, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5
    #plt.imshow(blur, cmap='gray'), plt.show()

    
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobelx + sobely)

    img_canny = cv2.Canny(sobel, threshold1=50, threshold2=120) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5)) #15-3
    closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    img_dil = cv2.dilate(opening, ee_dil)
    plt.imshow(img_dil, cmap='gray'), plt.show()

    img_rellena = rellenar(img_dil)
    plt.imshow(img_rellena, cmap='gray'), plt.show()




nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02']#,'06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.medianBlur(img_gray, 5)  # kernel 5 normalmente funciona muy bien

    # 3) Mejorar contraste — CLAHE
    #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    #gray_clahe = clahe.apply(gray)

    #gray_clahe = histograma(gray,31,31)

    gray_clahe = cv2.addWeighted(gray, 1.7, blur, -0.7, 0)

    edges = cv2.Canny(gray_clahe, 80, 200)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)



    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = img_rgb.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:     # filtro de ruido
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        ratio = w / float(h)

        if 2.0 < ratio < 5.0:     # proporción clásica de patente
            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)



    # Mostrar
    plt.imshow(img_copy, cmap='gray')
    plt.title("MedianBlur + CLAHE")
    plt.show()


nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02','06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(0.5*sobelx + 0.5*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (6,4))
    img_dil = cv2.dilate(sobel, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1)) #15-3
    closing = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    blur2 = cv2.GaussianBlur(opening, (3,3), 0) #5,5  1.5

    img_canny = cv2.Canny(blur2, threshold1=50, threshold2=120) #50-120
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5,2))
    img_dil2 = cv2.dilate(img_canny, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1)) #15-3
    closing2 = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    opening2 = cv2.morphologyEx(closing2, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    #elemento_estructural_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)) 
    #img_erosionada = cv2.erode(opening2, elemento_estructural_2, iterations=1) 
    #plt.imshow(img_erosionada, cmap='gray'), plt.show()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,4))
    opening3 = cv2.morphologyEx(img_erosionada, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening3, cmap='gray'), plt.show()

    rell = rellenar(img_erosionada)
    plt.imshow(rell, cmap='gray'), plt.show()

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img_rgb.copy()
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        a=cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)
    
    plt.imshow(img_copy), plt.show()
    





img_h1 = histograma(closing, 3, 3)
img_h2 = histograma(closing, 23, 23)
img_h3 = histograma(closing, 53, 53)
img_h4 = histograma(closing, 101, 101)

ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (6,4))
img_dil = cv2.dilate(img_h4, ee_dil)
#plt.imshow(img_dil, cmap='gray'), plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10)) #15-3
closing = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, kernel)
#plt.imshow(closing, cmap='gray'), plt.show()



plt.figure(figsize=(14, 6))
plt.title("Imagen con detalles escondidos"), plt.axis('off')
plt.subplot(141)
plt.imshow(img_h1, cmap='gray'), plt.title("Kernel = 3x3")
plt.subplot(142)
plt.imshow(img_h2, cmap='gray'), plt.title("Kernel = 23x23") #Estos parametros son los mas adecuados
plt.subplot(143)
plt.imshow(img_h3, cmap='gray'), plt.title("Kernel = 53x53")
plt.subplot(144)
plt.imshow(img_h4, cmap='gray'), plt.title("Kernel = 73x73")
plt.show()



##muy bien!!
nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
#nro_patentes = ['02','06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show()

    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(0.5*sobelx + 0.5*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    opening = cv2.morphologyEx(sobel, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
    img_dil2 = cv2.dilate(opening, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10)) #15-3
    closing = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    plt.imshow(closing, cmap='gray'), plt.show()

    #blur2 = cv2.GaussianBlur(closing, (3,3), 0) #5,5  1.5

    img_canny = cv2.Canny(closing, threshold1=60, threshold2=180) #50-120
    #plt.imshow(img_canny, cmap='gray'),plt.title(nro_patente), plt.show()


    for x in (1,2,3,4,5):
        ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img_dil3 = cv2.dilate(img_canny, ee_dil)
        #plt.imshow(img_dil3, cmap='gray'),plt.title(nro_patente) ,plt.show()

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,2))
        opening3 = cv2.morphologyEx(img_dil3, cv2.MORPH_OPEN, kernel)

        img_canny = opening3

    plt.imshow(opening3, cmap='gray'),plt.title(nro_patente), plt.show()


    img_rellena = rellenar(img_dil3)
    plt.imshow(img_rellena, cmap='gray'), plt.show()
    

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening3 = cv2.morphologyEx(img_dil3, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening3, cmap='gray'), plt.show()
    

nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
patentes_ok = ['01','04','05','06','07','08','09','10']
patentes_maso = ['02','03','04','10','11','12']
patentes_mal = ['03','11','12']
#nro_patentes = ['02','06']
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show()
    
    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(0.75*sobelx + 0.75*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5)) #15-3
    #closing = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    opening = cv2.morphologyEx(sobel, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
    img_dil2 = cv2.dilate(opening, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10)) #15-3
    closing = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show() 


    for x in (1,2,3):
        
        img_out = closing.astype(np.float32)

        # Menores a 50 → 0
        img_out[img_out < 50] = 0

        # El resto +30%
        img_out[img_out >= 130] *= 1.4
        img_out[img_out < 130] *= -1.4

        # Recortar entre 0 y 255
        img_out = np.clip(img_out, 0, 255)

        # Convertir de vuelta a enteros
        img_out = img_out.astype(np.uint8)

        closing = img_out
     
    #plt.imshow(img_out, cmap='gray'), plt.show()

    resultado = cv2.bitwise_and(img_rgb, img_rgb, mask=img_out)
    plt.imshow(resultado), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    img_dil2 = cv2.dilate(img_out, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,4)) #10,2
    closing = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show() 

    img_canny = cv2.Canny(closing, threshold1=80, threshold2=190) #50-120
    #plt.imshow(img_canny, cmap='gray'),plt.title(nro_patente), plt.show()

    img_rellena = rellenar(img_canny)
    #plt.imshow(img_rellena, cmap='gray'),plt.title(nro_patente), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (4,2))
    img_rellena = cv2.dilate(img_rellena, ee_dil)
    #plt.imshow(img_rellena, cmap='gray'), plt.show()

    contours, hierarchy = cv2.findContours(img_rellena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 

        aspect_ratio = w / float(h)

        if aspect_ratio>2: 
            r = cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255,0 , 0), 2)
            c = cv2.drawContours(img_draw, [cnt], -1, (0, 255, 0), 3)
            cv2.putText(img_draw, str(aspect_ratio), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    plt.imshow(img_draw),plt.title(nro_patente), plt.show()


nro_patente = '04'
nro_patentes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
patentes_ok = ['01','04','05','06','07','08','09','10']
patentes_maso = ['02','03','04','10','11','12']
patentes_mal = ['02','03','11']
patentes = []
for nro_patente in nro_patentes:
    path = 'imagenes\patentes\img'+ nro_patente +'.png'
    img      = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap='gray'), plt.show()
    
    blur = cv2.GaussianBlur(img_gray, (3,3), 0) #5,5  1.5

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(0.75*sobelx + 0.75*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    opening = cv2.morphologyEx(sobel, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
    img_dil2 = cv2.dilate(opening, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10)) #15-3
    closing = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show() 


    for x in (1,2,3):

        img_out = closing.astype(np.float32)

        # Menores a 50 → 0
        img_out[img_out < 50] = 0

        # El resto +30%
        img_out[img_out >= 130] *= 1.4
        img_out[img_out < 130] *= -1.4

        # Recortar entre 0 y 255
        img_out = np.clip(img_out, 0, 255)

        # Convertir de vuelta a enteros
        img_out = img_out.astype(np.uint8)

        closing = img_out
     
    #plt.imshow(img_out, cmap='gray'), plt.show()

    #resultado = cv2.bitwise_and(img_rgb, img_rgb, mask=img_out)
    #plt.imshow(resultado), plt.show()

    img_contornos_horizontales = np.zeros_like(img_out)   # Forma real
    img_bboxes_horizontales    = np.zeros_like(img_out)   # Rectángulos

    contours, hierarchy = cv2.findContours(img_out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if aspect_ratio >= 2 and aspect_ratio < 3.5 and area > 200:
            # Imagen A: agregar el contorno REAL
            a=cv2.drawContours(img_contornos_horizontales, [cnt], -1, 255, thickness=cv2.FILLED)
            print('Vehiculo: ',str(nro_patente), '- Ratio:', str(aspect_ratio),'Area: ', str(area) )
            # Imagen B: agregar el bounding box completo
            a=cv2.rectangle(img_bboxes_horizontales, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
        

    img_contornos_horizontales = (img_contornos_horizontales > 0).astype(np.uint8) * 255
    img_bboxes_horizontales    = (img_bboxes_horizontales > 0).astype(np.uint8) * 255
    #plt.imshow(img_contornos_horizontales, cmap='gray'), plt.show()
    
    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (50,25))
    img_dil2 = cv2.dilate(img_bboxes_horizontales, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    resultado = cv2.bitwise_and(img_rgb, img_rgb, mask=img_dil2)
    #plt.imshow(resultado),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    #TRABAJO SOBRE LA PATENTE

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (55,30))
    img_dil2 = cv2.dilate(img_contornos_horizontales, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()


    contours, hierarchy = cv2.findContours(img_dil2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 
        peri = cv2.arcLength(cnt, True)

        for factor in [0.01, 0.02, 0.03, 0.05, 0.1]:
            epsilon = factor * peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                break
        if len(approx) != 4:
            patentes.append([nro_patente,None])
            break

        pts = approx.reshape(4, 2)
        
        pts_ordenados = pts[np.argsort(pts[:,1])]

        arriba = pts_ordenados[:2]
        abajo  = pts_ordenados[2:]

        arriba = arriba[np.argsort(arriba[:,0])]
        abajo  = abajo[np.argsort(abajo[:,0])]

        # Asignamos puntos en orden
        tl = arriba[0]   # top-left
        tr = arriba[1]   # top-right
        bl = abajo[0]    # bottom-left
        br = abajo[1]    # bottom-right
        
        src_pts = np.float32([tl, tr, br, bl])

        w, h = 400, 220
        dst_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])

        H, _ = cv2.findHomography(src_pts, dst_pts)
        enderezada = cv2.warpPerspective(resultado, H, (w, h))
       
        patentes.append([nro_patente,enderezada])

        plt.imshow(enderezada),plt.title('Vehiculo: '+str(nro_patente)), plt.show()


for patente in patentes:
    if patente[1] is None:
        print('Patente del vehiculo nro '+patente[0]+' no detectada.')
        continue
    plt.imshow(patente[1]),plt.title('Vehiculo: '+patente[0]), plt.show()    



        pts = approx.reshape(4,2)
        s = pts.sum(axis=1)
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]

        src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)
        max_height = int(max(height_left, height_right))

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, H, (max_width, max_height))





    resultado_grey = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)


    resultado_grey = histograma(resultado_grey,15,51)
    #plt.imshow(resultado_grey, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    for x in (1,2,3):

        img_out = resultado_grey.astype(np.float32)

        # Menores a 50 → 0
        img_out[img_out < 50] = 0

        # El resto +30%
        img_out[img_out >= 120] *= 1.1
        img_out[img_out < 120] *= -1.1

        # Recortar entre 0 y 255
        img_out = np.clip(img_out, 0, 255)

        # Convertir de vuelta a enteros
        img_out = img_out.astype(np.uint8)
        resultado_grey = img_out

    #plt.imshow(resultado_grey, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    patente_umbralada = cv2.threshold(img_out, 220, 255, cv2.THRESH_BINARY)[1]
    #plt.imshow(patente_umbralada, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    opening = cv2.morphologyEx(patente_umbralada, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,1))
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()


    invertida = cv2.bitwise_not(resultado_grey)
    plt.imshow(invertida, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()


    contours, hierarchy = cv2.findContours(invertida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    img_rell = rellenar(resultado_grey)
    #plt.imshow(img_rell, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    img_dil2 = cv2.erode(resultado_grey, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()
    
    



    
    hist_gray = cv2.equalizeHist(resultado_grey)
    #plt.imshow(hist_gray, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    blur = cv2.GaussianBlur(hist_gray, (5,5), 1)
    #plt.imshow(hist_gray, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    patente_umbralada = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
    #plt.imshow(patente_umbralada, cmap='gray'),plt.title('Vehiculo: '+str(nro_patente)), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(patente_umbralada, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    #plt.imshow(opening, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    img_dil2 = cv2.erode(opening, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()


    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dil2) 

    elementos_patente = img_rgb.copy()

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cv2.rectangle(elementos_patente, (x, y), (x+w, y+h), (0,255,0), 2)

    plt.imshow(elementos_patente),plt.title('Vehiculo: '+str(nro_patente)), plt.show()


    



    plt.imshow(img_contornos_horizontales, cmap='gray'), plt.show()


        