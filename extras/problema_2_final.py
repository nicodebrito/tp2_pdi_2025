
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
    
    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (50,15))
    img_dil2 = cv2.dilate(img_bboxes_horizontales, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    resultado = cv2.bitwise_and(img_rgb, img_rgb, mask=img_dil2)
    plt.imshow(resultado),plt.title('Vehiculo: '+str(nro_patente)), plt.show()
