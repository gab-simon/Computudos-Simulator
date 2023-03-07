import cv2 as cv
import numpy as np
import random
import os

# Load template and figure
path_figure = './figures'
file = random.choice(os.listdir("./figures/"))
figure = cv.imread(os.path.join(path_figure,file))

path_model = './models'
file_model = random.choice(os.listdir("./models/"))
template = cv.imread(os.path.join(path_model,file_model))

# Load template and figure
# figure = cv.imread('./neia.jpg')
# template = cv.imread('./3098_green.png')

figure = cv.resize(figure, (template.shape[1], template.shape[0]))

# Set blank image
blank = template.copy()
blank[:,:] = (0,0,0)

# Set the lower and upper green
lower_green = np.array([23,255,17])
upper_green = np.array([23,255,17])

# Set the masks
mask = cv.inRange(template, lower_green, upper_green)
mask_inv = cv.bitwise_not(mask)

# Set contours
contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
cv.drawContours(contour, contours, 0, (255, 255, 255), 3)
x,y,w,h = cv.boundingRect(contour)
largura = w
altura = h

resized = cv.resize(figure, [largura, altura])

# Overlay blank image
print('Resized tamanho =', resized.shape[1], resized.shape[0])
print('Coordenadas do resized =', x, x+w, y, y+h)
print('Template tamanho =', template.shape[1], template.shape[0])
blank[y:y+h, x:x+w] = resized
cv.imwrite('./blank.png', blank)
print('template\n', template, '\nresized\n', resized)
# apply the mask to the foreground image
figure_masked = cv.bitwise_and(blank, blank, mask=mask)

# apply the inverted mask to the background image
template_masked = cv.bitwise_and(template, template, mask=mask_inv)

cv.imwrite('./contour.png', contour)

# combine the masked foreground and background images
result = cv.add(template_masked, figure_masked)
cv.imwrite('./result.png', result)
cv.waitKey(0)

