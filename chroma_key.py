import cv2 as cv
import numpy as np

# Load template and figure
figure = cv.imread('./neia.jpg')
template = cv.imread('./3098_green.png')

figure = cv.resize(figure, (template.shape[1], template.shape[0]))

blank = np.zeros((template.shape[1], template.shape[0]), np.uint8)

cv.imwrite('./blank.png', blank)

# Set the lower and upper green
lower_green = np.array([0, 235, 0])
upper_green = np.array([0, 235, 0])

# Set the masks
mask = cv.inRange(template, lower_green, upper_green)
mask_inv = cv.bitwise_not(mask)

contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour = np.zeros((mask.shape[0], mask.shape[1]), np.uint8)
cv.drawContours(contour, contours, 0, (255, 255, 255), 3)
x,y,w,h = cv.boundingRect(contour)
largura = x+w
altura = y+h

resized = cv.resize(figure, [largura, altura], interpolation=cv.INTER_AREA)
cv.imwrite('./resized.png', resized)

# apply the mask to the foreground image
figure_masked = cv.bitwise_and(figure, figure, mask=mask)

# apply the inverted mask to the background image
template_masked = cv.bitwise_and(template, template, mask=mask_inv)



print(x,w,y,h)
rectangle = cv.rectangle(contour, (x,y), (x+w, y+h), (255,255,255), 5)

cv.imwrite('./contour.png', rectangle)

# combine the masked foreground and background images
result = cv.add(template_masked, figure_masked)
cv.imwrite('./result.png', result)
cv.waitKey(0)

