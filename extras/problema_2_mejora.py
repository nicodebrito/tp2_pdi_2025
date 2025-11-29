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
    sobel = cv2.convertScaleAbs(0.75*sobelx + 0.75*sobely)
    #plt.imshow(sobel, cmap='gray'), plt.show()
    
    #plt.imshow(img_out, cmap='gray'), plt.show() 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5)) #15-3
    closing = cv2.morphologyEx(img_out, cv2.MORPH_CLOSE, kernel)
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
        img_out[img_out >= 190] *= 1.3
        img_out[img_out < 130] *= -1.4

        # Recortar entre 0 y 255
        img_out = np.clip(img_out, 0, 255)

        # Convertir de vuelta a enteros
        img_out = img_out.astype(np.uint8)

        closing = img_out
    #plt.imshow(img_out, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (15,5))
    img_dil2 = cv2.dilate(img_out, ee_dil)
    #plt.imshow(img_dil2, cmap='gray'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2)) #15-3
    closing = cv2.morphologyEx(img_dil2, cv2.MORPH_CLOSE, kernel)
    #plt.imshow(closing, cmap='gray'), plt.show() 

    img_canny = cv2.Canny(closing, threshold1=80, threshold2=190) #50-120
    #plt.imshow(img_canny, cmap='gray'),plt.title(nro_patente), plt.show()

    img_rellena = rellenar(img_canny)
    #plt.imshow(img_rellena, cmap='gray'),plt.title(nro_patente), plt.show()

    contours, hierarchy = cv2.findContours(img_rellena, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt) 

        aspect_ratio = w / float(h)

        if aspect_ratio>1.5: 
            r = cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255,0 , 0), 2)
            c = cv2.drawContours(img_draw, [cnt], -1, (0, 255, 0), 3)
    
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    plt.imshow(img_draw),plt.title(nro_patente), plt.show()





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


    #for x in (1,2,3):

    img_out = closing.astype(np.float32)
    # Menores a 50 → 0
    img_out[img_out < 50] = 0
    # El resto +30%
    img_out[img_out >= 130] = 255
    img_out[img_out < 130]  = 0
    # Recortar entre 0 y 255
    img_out = np.clip(img_out, 0, 255)
    # Convertir de vuelta a enteros
    img_out = img_out.astype(np.uint8)
    #closing = img_out
     
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
    plt.imshow(resultado),plt.title('Vehiculo: '+str(nro_patente)), plt.show()