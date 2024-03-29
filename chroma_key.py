import cv2 as cv
import sys
import numpy as np
import random
import getopt
import os

def getScriptArguments(argv):
    global set_figure
    global set_model
    arg_help = 'Comando incorreto.'
    
    try:
        opts, args = getopt.getopt(argv[1:], 'f:t:', ['figure', 'template'])
    except:
        print(arg_help) 
        sys.exit(2)
    
    # Treat script arguments
    for opt, arg in opts:
        if opt in ('-f', '--figure'):
            set_figure = arg
            if(os.path.exists(os.path.join('./',set_figure)) == False):
                print(f'Arquivo {set_figure} não existe!')
                quit()
        if opt in ('-t', '--template'):
            set_model = arg
            if(os.path.exists(os.path.join('./',set_model)) == False):
                print(f'Arquivo {set_figure} não existe!')
                quit()

set_figure = ''
set_model = ''
path_model = './models'
path_figure = './figures'

if __name__ == "__main__":
    getScriptArguments(sys.argv)

# Load template and figure
if set_figure == '':
    file = random.choice(os.listdir("./figures/"))
    figure = cv.imread(os.path.join(path_figure,file))
else:
    file = set_figure
    figure = cv.imread(file)
if set_model == '':
    file_model = random.choice(os.listdir("./models/"))
    template = cv.imread(os.path.join(path_model, file_model))
else:
    file_model = set_model
    template = cv.imread(file_model)

# Resize figure for image operations
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

# Find the size of the green box
x,y,w,h = cv.boundingRect(contour)
widht = w
height = h

# Resize the figure to box size
resized = cv.resize(figure, [widht, height])

# Overlay blank image
blank[y:y+h, x:x+w] = resized

# Apply the mask to the figure image
figure_masked = cv.bitwise_and(blank, blank, mask=mask)

# Apply the inverted mask to the template image
template_masked = cv.bitwise_and(template, template, mask=mask_inv)

# combine the masked figure and template images
result = cv.add(template_masked, figure_masked)
cv.imwrite('./result.png', result)
cv.waitKey(0)

