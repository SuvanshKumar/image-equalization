import cv2
import numpy as np
import sys
import math
import collections

# read arguments
if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

# check the correctness of the input parameters
if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

# read image
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()


# check for color image and change w1, w2, h1, h2 to pixel locations 
rows, cols, bands = inputImage.shape
if(bands != 3) :
    print("Input image is not a standard color image:", inputImage)
    sys.exit()

W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be applied only to
# the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

luv_img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2Luv)
window_height = H2 - H1 + 1
window_width = W2 - W1 + 1
window = luv_img[H1:H2+1, W1:W2+1].copy()
L_array = window[:,:,0]

num_of_pixels = 0
histogram ={}
num_of_values = 256

for i in range(L_array.shape[0]):
    for j in range(L_array.shape[1]):
        histogram[L_array[i][j]] = histogram.get(L_array[i][j], 0) + 1
        num_of_pixels += 1

histogram = collections.OrderedDict(sorted(histogram.items()))

f_i = 0

cumulative_hist = {}

for k, v in histogram.items():
    f_i += v
    cumulative_hist[k] = f_i

keys = list(histogram.keys())

replacements = {}

for i in range(len(keys)):
    if i == 0:
        f_iminus1 = 0
    else:
        f_iminus1 = cumulative_hist[keys[i-1]]
    f_i = cumulative_hist[keys[i]]
    floorval = math.floor((f_iminus1 + f_i) * (num_of_values) / (2 * num_of_pixels))
    replacements[keys[i]] = floorval

equalized_L = np.zeros([window_height, window_width], dtype=np.uint8)

for i in range(window_height):
    for j in range(window_width):
        equalized_L[i, j] = replacements[L_array[i, j]]

# equalized_L = cv2.equalizeHist(L_array)

window[:, :, 0] = equalized_L
luv_img[H1:H2+1, W1:W2+1] = window
outputImage = cv2.cvtColor(luv_img, cv2.COLOR_Luv2BGR)

# saving the output - save the gray window image
cv2.imwrite(name_output, outputImage)
