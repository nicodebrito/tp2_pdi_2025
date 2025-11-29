import matplotlib.pyplot as plt
import numpy as np
import cv2


def segmenta_patente(img):
    print('Inicia proceso de deteccion y segmentacion de patente')
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

    img_out = cv2.threshold(closing, 129, 255, cv2.THRESH_BINARY)[1]
    #plt.imshow(img_out, cmap='gray'), plt.show()

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
            #print('Vehiculo: ',str(nro_patente), '- Ratio:', str(aspect_ratio),'Area: ', str(area) )
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
            #patentes.append([nro_patente,None,None,None])
            return None
            #break

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

        w, h = 400, 175 #
        dst_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])

        H, _ = cv2.findHomography(src_pts, dst_pts)
        enderezada = cv2.warpPerspective(resultado, H, (w, h))
       
        enderezada1=enderezada[40:140,:]

        print('Finaliza proceso de deteccion y segmentacion de patente')
        return enderezada1             
        #patentes.append([nro_patente,enderezada1,resultado,img_rgb])


def identificar_caracteres(patente):
    print('Inicia deteccion de componentes de patente')
    if patente is None:
        print('No se detecta patente') 
        return None 

    img_rgb = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)
    #plt.imshow(patente), plt.show()

    # Convertir a HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Equalizar SOLO el canal V (aumenta contraste general)
    v_eq = cv2.equalizeHist(v)

    # Reensamblar
    hsv_eq = cv2.merge([h, s, v_eq])
    img_contraste = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    img_grey = cv2.cvtColor(img_contraste, cv2.COLOR_RGB2GRAY)

    blur_bg = cv2.GaussianBlur(img_grey, (71,25), 0)

    res = cv2.subtract(img_grey, blur_bg)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55,25))
    tophat = cv2.morphologyEx(res, cv2.MORPH_TOPHAT, kernel)
    #blackhat = cv2.morphologyEx(res, cv2.MORPH_BLACKHAT, kernel)

    img_umbralada = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #plt.imshow(img_umbralada, cmap= 'gray'), plt.show()

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbralada) 
    #output = img_rgb.copy()
    mask_final = np.zeros(labels.shape, dtype=np.uint8)
    img_out = patente.copy()
    
    H, W = img_umbralada.shape[:2]
    min_h = 0.20 * H
    max_h = 0.80 * H
    min_w = 0.05 * W
    max_w = 0.25 * W
    min_area = 0.01 * W * H     # 1% del area total
    max_area = 0.1 * W * H      # 10% del area total
    print('***************************')
    print('Caracteristicas imagen de patente')
    print('min_h: ', min_h ) 
    print('max_h: ', max_h  )
    print('min_w: ', min_w  )
    print('max_w: ', max_w  )
    print('min_area: ',min_area)
    print('max_area: ',max_area)
    
    elementos = []
    print('Inicia analisis de componentes detectadas')
    for i in range(1, n):  # 0 es fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        #area = stats[i, cv2.CC_STAT_AREA]
        area = w * h
        ratio = h / float(w)
        
        #Coordenadas centroide
        cx = int(x + w/2)
        cy = int(y + h/2)
        
        print('------------------------')
        print('Componente: ',str(i))
        print('Ratio: ',ratio)
        print('Ancho: ',w)
        print('Alto: ',h)
        print('Area: ',area)
        print('Ubicacion centroide: ',cx ,';',cy)

        if ratio < 1.0 or ratio > 4.0:
            restriccion = 'Ratio'
            print('Se descarta por: ',restriccion)
            continue
        if not (min_w <= w <= max_w): 
            restriccion = 'Ancho'
            print('Se descarta por: ',restriccion)
            continue
        if not (min_h <= h <= max_h): 
            restriccion = 'Alto'
            print('Se descarta por: ',restriccion)
            continue
        if not (min_area <= area <= max_area): 
            restriccion = 'Area'
            print('Se descarta por: ',restriccion)
            continue
        if (h <= w): 
            restriccion = 'h<=w'
            print('Se descarta por: ',restriccion)
            continue
        if (h > H*(2/3)): 
            restriccion = 'h > H*(2/3)'
            print('Se descarta por: ',restriccion)
            continue
        
        yaExiste = 0
        for elemento in elementos:
            cx2 = elemento[0]
            #cy2 = elemento[1]
            cw2 = elemento[2]
            dist_x = np.abs(x - cx2)
            if dist_x < (cw2/2 + w/2)*0.8:
                yaExiste = 1
                print('Elemento en conflicto')
                print('Distancia conflicto: ', dist_x,'. Es menor igual a ', str(0.8*cw2))
                break

        if yaExiste == 0 and len(elementos) <= 6:     
            elementos.append(stats[i])
            color = (0, 255, 0)
            a=cv2.rectangle(img_out, (x, y), (x+w, y+h), color, 1)
            #b=cv2.putText(img_out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            print('SE GRAFICA')
        else:
            restriccion = 'exceso de elementos graficados.'
            print('Se descarta por: ',restriccion)

    #final.append([p[0],p[3],img_out])
    return img_out

def visualizar_resultados(resultados):
    
    placeholder = np.full((100, 400), 150, dtype=np.uint8)
    #Los elementos de resultado son [id_img, img_rgb, patente, patente_segmentada]    
    for nro_vehiculo, v, p, patente_segmentada in resultados:

        plt.figure(figsize=(12,5))
        plt.suptitle('Imagen: ' + str(nro_vehiculo))

        # Vehiculo
        plt.subplot(1, 3, 1)
        img_v = v if v is not None else placeholder
        plt.imshow(img_v, cmap='gray')
        if v is None:
            plt.text(0.5, 0.5, "Sin imagen", ha='center', va='center', fontsize=12,
                     transform=plt.gca().transAxes)
        plt.title("Vehículo")
        plt.axis('off')

        # Patente
        plt.subplot(1, 3, 2)
        img_p = p if p is not None else placeholder
        plt.imshow(img_p, cmap='gray', vmin=0, vmax=255)
        if p is None:
            plt.text(0.5, 0.5, "Patente\nNO detectada", ha='center', va='center',
                     fontsize=12, transform=plt.gca().transAxes)
        plt.title("Patente")
        plt.axis('off')

        # Caracteres
        plt.subplot(1, 3, 3)
        img_ps = patente_segmentada if patente_segmentada is not None else placeholder
        plt.imshow(img_ps, cmap='gray', vmin=0, vmax=255)
        if patente_segmentada is None:
            plt.text(0.5, 0.5, "Sin elementos", ha='center', va='center', fontsize=12,
                     transform=plt.gca().transAxes)
        plt.title("Caracteres detectados")
        plt.axis('off')

        plt.tight_layout()
        plt.show()



id_imagenes = ['01','02','03', '04','05','06','07','08','09','10','11','12']
vehiculos = []
patentes = []
patentes_segmentadas = []
resultados = []

for id_img in id_imagenes:
    
    #Accedo a la imagen
    path = 'imagenes\patentes\img'+ id_img +'.png'
    img  = cv2.imread(path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vehiculos.append(img_rgb)
    print('Vehiculo:',id_img)
    
    #Segmento patente
    patente = segmenta_patente(img)
    #plt.imshow(patente), plt.show()
    patentes.append(patente)

    #Identifico caracteres
    patente_segmentada = identificar_caracteres(patente)
    #plt.imshow(patente_segmentada), plt.show()
    patentes_segmentadas.append(patente_segmentada)
    
    #Guardo los resultados para visualizar en conjunto
    resultados.append([id_img,img_rgb,patente,patente_segmentada])


visualizar_resultados(resultados)
