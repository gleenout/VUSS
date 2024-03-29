import cv2
import numpy as np

# Importa módulo para abrir caixa de seleção de arquivos
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Abre caixa de seleção de arquivos
Tk().withdraw()
filename = askopenfilename()

# Lê imagem a partir do arquivo selecionado
img = cv2.imread(filename)

# Fator de calibração
FPX = 0.04

# Converte para grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica threshold adaptativo para segmentar úlcera
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 3)

# Converte para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold no canal H (Hue)
lower = np.array([0, 50, 50])
upper = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

# Máscara por textura com Laplaciano
#lap = cv2.Laplacian(gray, cv2.CV_64F)
#lap = np.uint8(np.absolute(lap))
#mask_lap = cv2.threshold(lap, 12, 255, cv2.THRESH_BINARY)[1]

# Combina máscaras
#mask = mask_lap & mask_lap

# Operações morfológicas para limpar mascaras
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Encontra contornos
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [np.array(cnt).astype(np.int32) for cnt in contours]

# Filtra por área para pegar contorno maior
#ulcer_contour = max(contours, key=cv2.contourArea)

# Calcula área, exibe resultado
#area_px = cv2.contourArea(ulcer_contour)
#area_cm = area_px * FPX * FPX

# Desenha contornos
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)

# Maior contorno
max_c = max(contours, key=cv2.contourArea)
area_px = cv2.contourArea(max_c)
area_cm = area_px * FPX * FPX

# Encontra contorno convex hull
hull = cv2.convexHull(contours[0])

# Simplifica contorno para reduzir pontos
hull = cv2.approxPolyDP(hull, 0.01*cv2.arcLength(hull,True), True)

# Cria matriz de altura com dimensões da imagem
height_map = np.zeros(thresh.shape)

# Preenche matriz com distância euclidiana de cada pixel até o contorno
for y in range(height_map.shape[0]):
    for x in range(height_map.shape[1]):
        dist = cv2.pointPolygonTest(hull, (x,y), True)
        height_map[y,x] = dist

# Normaliza alturas de 0 a 255
height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
height_map = (height_map * 255).astype(np.uint8)

# Gera malha de pixels
x,y = np.mgrid[0:height_map.shape[0], 0:height_map.shape[1]]

# Exibe resultado
print(f'Área: {area_cm:.2f} cm2')
cv2.imshow('Contornos', contour_img)
cv2.imshow('Gray', gray)
cv2.imshow('Thresh', thresh)
cv2.imshow('HSV', hsv)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()


def main():
    pass


if __name__ == "__main__":
    main()