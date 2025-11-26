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

        w, h = 400, 175 #
        dst_pts = np.float32([[0,0], [w,0], [w,h], [0,h]])

        H, _ = cv2.findHomography(src_pts, dst_pts)
        enderezada = cv2.warpPerspective(resultado, H, (w, h))
       
        enderezada1=enderezada[40:140,:]

        patentes.append([nro_patente,enderezada1,resultado])

        plt.imshow(enderezada1),plt.title('Vehiculo: '+str(nro_patente)), plt.show()


for p in patentes:
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(p[2], cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(p[1], cmap='gray')
    plt.axis('off')

    plt.show()


p = patentes[5]
patentes.pop()

for p in patentes:
    #plt.imshow(p[1]),plt.title('Vehiculo: '+str(p[0])), plt.show()
    patente = p[1]
    
    #v_blur = cv2.medianBlur(patente,7)

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

    #enhanced = cv2.add(res, tophat)
    #enhanced = cv2.subtract(enhanced, blackhat)

    #plt.imshow(tophat, cmap= 'gray'), plt.show()

    #img_blur = cv2.GaussianBlur(img_grey,(3,3),0)

    #img_grey = cv2.equalizeHist(img_grey)

    #img_umbralada = img_grey.copy()
    #img_umbralada[img_umbralada<q1] = 0
    #img_umbralada[(img_umbralada >= q1) & (img_umbralada <= q2)] = q2
    #img_umbralada[img_umbralada>q3] = 255

    img_umbralada = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #th = cv2.adaptiveThreshold(
    #    img_blur, 255,
    #    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #    cv2.THRESH_BINARY_INV,
    #    41, 5
    #)


    #plt.imshow(img_umbralada, cmap= 'gray'), plt.show()

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(img_umbralada) 
    #output = img_rgb.copy()
    mask_final = np.zeros(labels.shape, dtype=np.uint8)
    img_out = patente.copy()
    print('Patente: ',p[0])
    H, W = img_umbralada.shape[:2]
    min_h = 0.20 * H
    max_h = 0.80 * H
    min_w = 0.05 * W
    max_w = 0.25 * W
    min_area = 0.01 * W * H   # 1% del área total
    max_area = 0.1 * W * H   # 20% del área total
    print()
    print('min_h: ', min_h ) 
    print('max_h: ', max_h  )
    print('min_w: ', min_w  )
    print('max_w: ', max_w  )
    print('min_area: ',min_area)
    print('max_area: ',max_area)
    print('***************************')
    elementos = []
    for i in range(1, n):  # 0 es fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        #area = stats[i, cv2.CC_STAT_AREA]
        area = w * h
        ratio = h / float(w)
        
        print('------------------------')
        print('Componente: ',str(i))
        print('Ratio: ',ratio)
        print('Ancho: ',w)
        print('Alto: ',h)
        print('Area: ',area)
        
        if ratio < 1.0 or ratio > 4.0:continue
        if not (min_w <= w <= max_w): continue
        if not (min_h <= h <= max_h): continue
        if not (min_area <= area <= max_area): continue
        if (h <= w): continue
        if (h > H*(2/3)): continue
    
        elementos.append(stats[i])
        if len(elementos) <= 6:
            a=cv2.rectangle(img_out, (x, y), (x+w, y+h), (0, 255, 0), 1)
            print('SE GRAFICA')
            print('------------------------')
            
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(patente)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_out)
    plt.axis('off')

    plt.show()


    plt.imshow(img_out), plt.show()



    img_contraste = cv2.equalizeHist(img_umbralada)

    mask1 = cv2.inRange(img_contraste,150,220)

    img_umbralada = cv2.threshold(img_contraste, 220, 255, cv2.THRESH_BINARY)[1]

    fusion = cv2.addWeighted(mask1, 0.2, img_umbralada, 0.8, 0)

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fusion,cmap= 'gray' )
    plt.axis('off')

    plt.show()




for p in patentes:
    #plt.imshow(p[1]),plt.title('Vehiculo: '+str(p[0])), plt.show()
    patente = p[1]

    v_blur = cv2.GaussianBlur(patente, (3,3), 0)
    
    #v_blur = cv2.medianBlur(patente,7)

    img_rgb = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)

    # Convertir a HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Equalizar SOLO el canal V (aumenta contraste general)
    v_eq = cv2.equalizeHist(v)

    # Reensamblar
    hsv_eq = cv2.merge([h, s, v_eq])
    img_contraste = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

    img_grey = cv2.cvtColor(img_contraste, cv2.COLOR_RGB2GRAY)

    img_grey = cv2.equalizeHist(img_grey)

    img_umbralada = img_grey.copy()
    img_umbralada[img_umbralada<q1] = 0
    img_umbralada[(img_umbralada >= q1) & (img_umbralada <= q2)] = q2
    img_umbralada[img_umbralada>q3] = 255

    #img_umbralada = cv2.threshold(img_umbralada, 200, 255, cv2.THRESH_BINARY)[1]
    th = cv2.adaptiveThreshold(
        img_umbralada, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 5
    )

    plt.imshow(th, cmap= 'gray'), plt.show()



    img_contraste = cv2.equalizeHist(img_umbralada)

    mask1 = cv2.inRange(img_contraste,150,220)

    img_umbralada = cv2.threshold(img_contraste, 220, 255, cv2.THRESH_BINARY)[1]

    fusion = cv2.addWeighted(mask1, 0.2, img_umbralada, 0.8, 0)

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fusion,cmap= 'gray' )
    plt.axis('off')

    plt.show()





    #plt.imshow(patente, cmap='gray'), plt.show()
    patente_grey = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)
    #plt.imshow(patente_grey, cmap='gray'), plt.show()

    patente_grey = cv2.equalizeHist(patente_grey)
    #plt.imshow(patente_grey, cmap='gray'), plt.show()

    patente_grey = patente_grey.astype(np.float32)

    patente_grey[patente_grey >200] = 255
    patente_grey[patente_grey <150] = 0
    patente_grey[patente_grey >= 150] *= 1.2
    patente_grey[patente_grey < 150] *= -1.2

    patente_grey = np.clip(patente_grey, 0, 255)
    patente_grey = patente_grey.astype(np.uint8)

    #plt.imshow(patente_grey, cmap='gray'), plt.show()

    img_umbralada = cv2.threshold(patente_grey, 220, 255, cv2.THRESH_BINARY)[1]
    #plt.imshow(img_umbralada, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (8,2))
    img_dil2 = cv2.dilate(img_umbralada, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,6))
    opening = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()



    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))
    opening = cv2.morphologyEx(img_umbralada, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()
    img_umbralada = cv2.threshold(patente_grey, 110, 255, cv2.THRESH_BINARY)[1]
    #plt.imshow(img_umbralada, cmap='gray'), plt.show()

    elemento_estructural_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2))
    img_erosionada = cv2.erode(img_umbralada, elemento_estructural_2, iterations=1) 
    #plt.imshow(img_erosionada, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(img_erosionada, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()



#hasta aca!

    #patente_eq = cv2.equalizeHist(patente_grey)
    #plt.imshow(patente_eq, cmap='gray'), plt.show()
    patente_eq = patente_grey
    for x in (1,2,3):

        img_out = patente_eq.astype(np.float32)

        # Menores a 50 → 0
        img_out[img_out < 40] = 0

        # El resto +30%
        img_out[img_out >= 150] *= 1.2
        img_out[img_out < 150] *= -1.2

        # Recortar entre 0 y 255
        img_out = np.clip(img_out, 0, 255)

        # Convertir de vuelta a enteros
        img_out = img_out.astype(np.uint8)

        patente_eq = img_out

    #plt.imshow(patente_eq, cmap='gray'), plt.show()

    elemento_estructural_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_erosionada = cv2.erode(patente_eq, elemento_estructural_2, iterations=1) 
    #plt.imshow(img_erosionada, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    opening = cv2.morphologyEx(img_erosionada, cv2.MORPH_OPEN, kernel)
    plt.imshow(opening, cmap='gray'), plt.show()

    img_umbralada = cv2.threshold(img_erosionada, 120, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(img_umbralada, cmap='gray'), plt.show()

    





