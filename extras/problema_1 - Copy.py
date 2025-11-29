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


# Carga de imagenes
img      = cv2.imread('imagenes/monedas.jpg')
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(img_rgb, cmap='gray'), plt.show()
plt.imshow(img_gray, cmap='gray'), plt.show()

#COMBO 1
img_blur  = cv2.GaussianBlur(img_gray, (25, 25),0)
plt.imshow(img_blur, cmap='gray'), plt.show()

img_canny = cv2.Canny(img_blur, threshold1=10, threshold2=17)
plt.imshow(img_canny, cmap='gray'), plt.show()

ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
img_dil = cv2.dilate(img_canny, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
AClose = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()

img_rellena = rellenar(AClose)
plt.imshow(img_rellena, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65,65))
Ao = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, B)
plt.imshow(Ao, cmap='gray'), plt.show()

elemento_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img_er = cv2.erode(Ao, elemento_erosion)
plt.imshow(img_er, cmap='gray'), plt.show()


n, labels, stats, _ = cv2.connectedComponentsWithStats(img_er)

img_vis = img.copy()
for i in range(1, n):  # salteamos el fondo
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Componentes detectados")
plt.show()


RHO_TH = 0.8
monedas = []
dados = []
rhos = []

for i in range(1,n):
    label = (labels == i).astype('uint8') * 255
    #plt.imshow(label, cmap='gray'), plt.show()
    ext_cont, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_cont[0])
    perimeter = cv2.arcLength(ext_cont[0], True)
    rho = 4 * np.pi * area /(perimeter ** 2)
    if rho >= RHO_TH:
        #masc_monedas[obj == 255,] = MONEDAS_C
        monedas.append(stats[i])
    else:
        #masc_dados[obj == 255,] = DADOS_C
        dados.append(stats[i])


img_vis = img.copy()
for m in monedas:  
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
for d in dados:  
    x = d[0]
    y = d[1]
    w = d[2]
    h = d[3]
    #Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Elementos detectados: dados y monedas")
plt.show()

img      = cv2.imread('imagenes/monedas.jpg')
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_clasificada = img_rgb.copy()
monedas_10 = 0
monedas_1 = 0
monedas_50 = 0
max_area_moneda = max(moneda[4] for moneda in monedas)
for m in monedas:
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]

    if max_area_moneda*0.95 < m[4] <= max_area_moneda: #Si el area es igual al area maxima o, a lo sumo, un 95% de ella, es una moneda grande.
        monedas_50+=1
        color = (255, 255, 255)
        texto = '50'
    
    elif max_area_moneda*0.8 < m[4] <= max_area_moneda*0.95: 
        monedas_1+=1
        color = (0, 255, 0)
        texto = '1'
    
    elif m[4] <= max_area_moneda*0.8:
        monedas_10+=1
        color = (255, 0, 0) 
        texto = '10'  

    cv2.rectangle(img_clasificada, (x, y), (x + w, y + h), color, 2)
    b=cv2.putText(img_clasificada, texto, (x, y-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_clasificada)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()



img_bgr_dados  = img.copy()
#img_gray_dados = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for dado in dados:
    x = dado[0]
    y = dado[1]
    w = dado[2]
    h = dado[3]
    recorte_dado = img_gray[y:y + w, x: x + h]
    recorte_dado = cv2.medianBlur(recorte_dado, 7)
    plt.imshow(recorte_dado, cmap='gray'), plt.show()
    circles = cv2.HoughCircles(recorte_dado,
                              cv2.HOUGH_GRADIENT,
                              1, 20,
                              param1=50, param2=50,
                              minRadius=20, maxRadius=50)
    n = 0
    if isinstance(circles, np.ndarray):
      n = len(circles[0])


    texto = str(n)
    cv2.putText(img_bgr_dados, texto, 
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                color=(255, 255, 255),
                thickness=20)

img_rgb_dados  = cv2.cvtColor(img_bgr_dados, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_dados)
plt.axis("off")
plt.title("Conteo de dados")
plt.show()









#FIN

#COMBO 2
# Carga de imagenes
img      = cv2.imread('imagenes/monedas.jpg')
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

plt.imshow(img_rgb, cmap='gray'), plt.show()
plt.imshow(img_gray, cmap='gray'), plt.show()

img_blur  = cv2.GaussianBlur(img_gray, (25, 25),0)
plt.imshow(img_blur, cmap='gray'), plt.show()

img_canny = cv2.Canny(img_blur, threshold1=10, threshold2=17)
plt.imshow(img_canny, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,35))
AClose = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()

ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,10))
img_dil = cv2.dilate(AClose, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

img_rellena = rellenar(img_dil)
plt.imshow(img_rellena, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (90,90)) #80-55
Ao = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, B)
plt.imshow(Ao, cmap='gray'), plt.show()



n, labels, stats, _ = cv2.connectedComponentsWithStats(Ao)

img_vis = img.copy()
for i in range(1, n):  # salteamos el fondo
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)


# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Componentes detectados")
plt.show()


RHO_TH = 0.85
monedas = []
dados = []
monedas_rhos = []
dados_rhos = []

for i in range(1,n):
    label = (labels == i).astype('uint8') * 255
    #plt.imshow(label, cmap='gray'), plt.show()
    ext_cont, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_cont[0])
    perimeter = cv2.arcLength(ext_cont[0], True)
    rho = 4 * np.pi * area /(perimeter ** 2)
    if rho >= RHO_TH:
        #masc_monedas[obj == 255,] = MONEDAS_C
        monedas.append(stats[i])
        monedas_rhos.append(rho)
    else:
        #masc_dados[obj == 255,] = DADOS_C
        dados.append(stats[i])
        dados_rhos.append(rho)


img_vis = img.copy()
for m in monedas:  
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
for d in dados:  
    x = d[0]
    y = d[1]
    w = d[2]
    h = d[3]
    #Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Elementos detectados: dados y monedas")
plt.show()

img      = cv2.imread('imagenes/monedas.jpg')
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_clasificada = img_rgb.copy()
monedas_10 = 0
monedas_1 = 0
monedas_50 = 0
max_area_moneda = max(moneda[4] for moneda in monedas)
for m in monedas:
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]

    if max_area_moneda*0.95 < m[4] <= max_area_moneda: #Si el area es igual al area maxima o, a lo sumo, un 95% de ella, es una moneda grande.
        monedas_50+=1
        color = (255, 255, 255)
    
    elif max_area_moneda*0.8 < m[4] <= max_area_moneda*0.95: 
        monedas_1+=1
        color = (0, 255, 0)
    
    elif m[4] <= max_area_moneda*0.8:
        monedas_10+=1
        color = (255, 0, 0)   

    cv2.rectangle(img_clasificada, (x, y), (x + w, y + h), color, 2)

# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_clasificada)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()


img_bgr_dados  = img.copy()
#img_gray_dados = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for dado in dados:
    x = dado[0]
    y = dado[1]
    w = dado[2]
    h = dado[3]
    recorte_dado = img_gray[y:y + w, x: x + h]
    recorte_dado = cv2.medianBlur(recorte_dado, 7)
    plt.imshow(recorte_dado, cmap='gray'), plt.show()
    circles = cv2.HoughCircles(recorte_dado,
                              cv2.HOUGH_GRADIENT,
                              1, 20,
                              param1=50, param2=50,
                              minRadius=20, maxRadius=50)
    n = 0
    if isinstance(circles, np.ndarray):
      n = len(circles[0])


    texto = str(n)
    cv2.putText(img_bgr_dados, texto, 
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                color=(255, 255, 255),
                thickness=20)

img_rgb_dados  = cv2.cvtColor(img_bgr_dados, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_dados)
plt.axis("off")
plt.title("Conteo de dados")
plt.show()





#FIN 

img_rellena = rellenar(img_dil)
plt.imshow(img_rellena, cmap='gray'), plt.show()

img_blur  = cv2.GaussianBlur(img_gray, (21, 21),0)
plt.imshow(img_blur, cmap='gray'), plt.show()

img_blur  = cv2.GaussianBlur(img_blur, (5, 5),0)
plt.imshow(img_blur, cmap='gray'), plt.show()

img_canny = cv2.Canny(img_blur, threshold1=15, threshold2=30)
plt.imshow(img_canny, cmap='gray'), plt.show()

ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,30))
img_dil = cv2.dilate(img_canny, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,25))
AClose = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()



img_blur  = cv2.GaussianBlur(img_gray, (5, 5),2)
plt.imshow(img_blur, cmap='gray'), plt.show()
img_canny = cv2.Canny(img_blur, 90, 190)
plt.imshow(img_canny, cmap='gray'), plt.show()


ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (42, 32))
img_dil = cv2.dilate(img_canny, ee_dil)
plt.imshow(img_dil, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,15))
Ao = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, B)
plt.imshow(Ao, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,20))
AClose = cv2.morphologyEx(Ao, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()



B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
Ao = cv2.morphologyEx(img_dil, cv2.MORPH_OPEN, B)
plt.imshow(Ao, cmap='gray'), plt.show()

ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 5))
img_dil_2 = cv2.dilate(Ao, ee_dil)
plt.imshow(img_dil_2, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
AClose = cv2.morphologyEx(img_dil_2, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()

ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
img_dil_2 = cv2.dilate(AClose, ee_dil)
plt.imshow(img_dil_2, cmap='gray'), plt.show()

img_rellena = rellenar(AClose)
plt.imshow(img_rellena, cmap='gray'), plt.show()







elemento_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
img_er = cv2.erode(img_rellena, elemento_erosion)
plt.imshow(img_er, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
AoP = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, B)
plt.imshow(AoP, cmap='gray'), plt.show()

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
AClose = cv2.morphologyEx(img_er, cv2.MORPH_CLOSE, B)
plt.imshow(AClose, cmap='gray'), plt.show()


#B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50,50))
#Aclau = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, B)
#plt.imshow(Aclau, cmap='gray'), plt.show()

#ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))
#img_dil = cv2.dilate(Aclau, ee_dil)
#plt.imshow(img_dil, cmap='gray'), plt.show()

#B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
#Aop = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, B)
#plt.imshow(Aop, cmap='gray'), plt.show()

n, labels, stats, _ = cv2.connectedComponentsWithStats(AoP)

img_vis = img.copy()
for i in range(1, n):  # saltar background
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Dibujar centroide
    #cx, cy = centroids[i]
    #cv2.circle(img_vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)

# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()

area_min = 9000
mascara = stats[:, cv2.CC_STAT_AREA] >= area_min

# Filtrar stats y centroids
stats_filtrados = stats[mascara]
#centroids_filtrados = centroids[mascara]


img_vis = img.copy()
for i in range(1, len(stats_filtrados)):  # saltar background
    x = stats_filtrados[i, cv2.CC_STAT_LEFT]
    y = stats_filtrados[i, cv2.CC_STAT_TOP]
    w = stats_filtrados[i, cv2.CC_STAT_WIDTH]
    h = stats_filtrados[i, cv2.CC_STAT_HEIGHT]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Dibujar centroide
    #cx, cy = centroids[i]
    #cv2.circle(img_vis, (int(cx), int(cy)), 3, (0, 255, 0), -1)

# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()



RHO_TH = 0.8
monedas = []
dados = []
rhos = []

for i in range(1,n):
    label = (labels == i).astype('uint8') * 255
    #plt.imshow(label, cmap='gray'), plt.show()
    ext_cont, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_cont[0])
    perimeter = cv2.arcLength(ext_cont[0], True)
    rho = 4 * np.pi * area /(perimeter ** 2)
    if rho >= RHO_TH:
        #masc_monedas[obj == 255,] = MONEDAS_C
        monedas.append(stats[i])
    else:
        #masc_dados[obj == 255,] = DADOS_C
        dados.append(stats[i])


img_vis = img.copy()
for m in monedas:  
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]

    # Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
for d in dados:  
    x = d[0]
    y = d[1]
    w = d[2]
    h = d[3]
    #Dibujar bounding box
    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
  
# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_vis)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()

img      = cv2.imread('imagenes/monedas.jpg')
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_clasificada = img_rgb.copy()
monedas_10 = 0
monedas_1 = 0
monedas_50 = 0
for m in monedas:
    x = m[0]
    y = m[1]
    w = m[2]
    h = m[3]
    if 60000 < m[4] < 70000:
        monedas_10+=1
        color = (255, 0, 0)        
    elif 90000 < m[4] < 96000:
        monedas_1+=1
        color = (0, 255, 0)
    else:            
        monedas_50+=1
        color = (255, 255, 255)

    cv2.rectangle(img_clasificada, (x, y), (x + w, y + h), color, 2)

# Mostrar resultado
plt.figure(figsize=(8, 8))
plt.imshow(img_clasificada)
plt.axis("off")
plt.title("Bounding Boxes en Aclau")
plt.show()



img_bgr_dados  = img.copy()
#img_gray_dados = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

for dado in dados:
    x = dado[0]
    y = dado[1]
    w = dado[2]
    h = dado[3]
    recorte_dado = img_gray[y:y + w, x: x + h]
    recorte_dado = cv2.medianBlur(recorte_dado, 7)
    circles = cv2.HoughCircles(recorte_dado,
                              cv2.HOUGH_GRADIENT,
                              1, 20,
                              param1=50, param2=50,
                              minRadius=20, maxRadius=50)
    n = 0
    if isinstance(circles, np.ndarray):
      n = len(circles[0])


    texto = str(n)
    cv2.putText(img_bgr_dados, texto, 
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                color=(255, 255, 255),
                thickness=20)

img_rgb_dados  = cv2.cvtColor(img_bgr_dados, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 8))
plt.imshow(img_rgb_dados)
plt.axis("off")
plt.title("Conteo de dados")
plt.show()

