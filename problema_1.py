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

def generar_mascara_identificacion(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_blur  = cv2.GaussianBlur(img_gray, (25, 25),0)
    #plt.imshow(img_blur, cmap='gray'), plt.show()

    img_canny = cv2.Canny(img_blur, threshold1=10, threshold2=17)
    #plt.imshow(img_canny, cmap='gray'), plt.show()

    ee_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    img_dil = cv2.dilate(img_canny, ee_dil)
    #plt.imshow(img_dil, cmap='gray'), plt.show()

    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    AClose = cv2.morphologyEx(img_dil, cv2.MORPH_CLOSE, B)
    #plt.imshow(AClose, cmap='gray'), plt.show()

    img_rellena = rellenar(AClose)
    #plt.imshow(img_rellena, cmap='gray'), plt.show()

    B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65,65))
    Ao = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, B)
    #plt.imshow(Ao, cmap='gray'), plt.show()

    elemento_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_er = cv2.erode(Ao, elemento_erosion)
    #plt.imshow(img_er, cmap='gray'), plt.show()

    print('Se ha generado la mascara para identificar los elementos.')

    return img_er

#n, labels, stats, _ = cv2.connectedComponentsWithStats(img_er)
#
#img_vis = img.copy()
#for i in range(1, n):  # salteamos el fondo
#    x = stats[i, cv2.CC_STAT_LEFT]
#    y = stats[i, cv2.CC_STAT_TOP]
#    w = stats[i, cv2.CC_STAT_WIDTH]
#    h = stats[i, cv2.CC_STAT_HEIGHT]
#
#    # Dibujar bounding box
#    cv2.rectangle(img_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#
## Mostrar resultado
#plt.figure(figsize=(8, 8))
#plt.imshow(img_vis)
#plt.axis("off")
#plt.title("Componentes detectados")
#plt.show()

def identificar_elementos(img, mask):

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    RHO_TH = 0.8
    monedas = []
    dados = []
    #rhos = []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

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

    img_vis = img_rgb.copy()
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

    return monedas,dados


def clasificar_monedas(img, monedas):

    img_clasificada = img.copy()
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
            color = (0, 0, 255) 
            texto = '10'  

        a=cv2.rectangle(img_clasificada, (x, y), (x + w, y + h), color, 2)
        b=cv2.putText(img_clasificada, texto, (x, y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
    
    img_rgb  = cv2.cvtColor(img_clasificada, cv2.COLOR_BGR2RGB)

    total_pesos = monedas_1+ 0.5*monedas_50 + 0.1*monedas_10
    total_monedas = monedas_1 + monedas_50 + monedas_10
    print('*****************************************************')
    print('Conteo de monedas')
    print('*****************************************************')
    print('Se contabiliza un total de',total_monedas,'monedas.')
    print('Monedas de $1:',monedas_1, ' unidades.' )
    print('Monedas de $0.50:',monedas_50,' unidades.' )
    print('Monedas de $0.10:',monedas_10,' unidades.' )
    print('*****************************************************')
    print('Las monedas suman un total de $',total_pesos)
    print('*****************************************************')
    
    # Mostrar resultado
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Clasificacion de monedas")
    plt.show()



def detectar_valores_dados(img, dados):
    img_bgr_dados  = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    valores_dados = [] 
    cantidad_dados = len(dados)
    for dado in dados:
        x = dado[0]
        y = dado[1]
        w = dado[2]
        h = dado[3]
        recorte_dado = img_gray[y:y + w, x: x + h]
        recorte_dado = cv2.medianBlur(recorte_dado, 7)
        #plt.imshow(recorte_dado, cmap='gray'), plt.show()
        circles = cv2.HoughCircles(recorte_dado,
                                  cv2.HOUGH_GRADIENT,
                                  1, 20,
                                  param1=50, param2=50,
                                  minRadius=20, maxRadius=50)
        n = 0
        if isinstance(circles, np.ndarray):
          n = len(circles[0])

        valores_dados.append(n)

        texto = str(n)
        a=cv2.rectangle(img_bgr_dados, (x, y), (x + w, y + h), (255, 255, 255), 2)
        b=cv2.putText(img_bgr_dados, texto, (x, y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
    
    suma_total_dados = sum(valores_dados)
    print('*****************************************************')
    print('Conteo de dados')
    print('*****************************************************')
    print('Se contabiliza un total de',cantidad_dados,'dados.')
    for n in range(cantidad_dados):
        print(f'Valor dado {str(n+1)}: {valores_dados[n]}' )
    print('*****************************************************')
    print('Los dados suman un total de',str(suma_total_dados))
    print('*****************************************************')
    
    img_rgb_dados  = cv2.cvtColor(img_bgr_dados, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb_dados)
    plt.axis("off")
    plt.title("Conteo de dados")
    plt.show()


#Inicio de proceso

img  = cv2.imread('imagenes/monedas.jpg')
mask = generar_mascara_identificacion(img)
monedas, dados = identificar_elementos(img, mask)
clasificar_monedas(img, monedas)
detectar_valores_dados(img, dados)