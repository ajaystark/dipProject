#Ajay Prakash , 2018215
import cv2
import numpy as np
from numpy.fft import fft2, ifft2


def get_density(image_name):
    image = cv2.imread(image_name,0)

    img=image



    size = int(5) // 2
    sigma=1
    normal = 1 / (2.0 * np.pi * sigma**2)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    gaussian_kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    
    img= cv2.filter2D(src=img, kernel=gaussian_kernel, ddepth=-1)

    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    angle=0
    Iy = cv2.filter2D(src=img, kernel=Ky, ddepth=-1)

    angle=180
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    theta=0
    Ix = cv2.filter2D(src=img, kernel=Kx, ddepth=-1)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    theta = np.arctan2(Iy, Ix)

    angle = theta * 180. / np.pi

    # print(theta)


    img=G

    cv2.imwrite('x gradient.jpg',Ix)
    cv2.imwrite('y gradient.jpg',Iy)

    angle[angle < 0] += 180

    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)

    def return_value(img,i,j):
        return img[i,j]

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            a=5
            p=100
            r = 255
            if (67.5 <= return_value(angle,i,j) < 112.5):
                q = return_value(img,i+1, j)
                r = return_value(img,i-1, j)
            elif (0 <= return_value(angle,i,j) < 22.5) or (157.5 <= return_value(angle,i,j) <= 180):
                q = return_value(img,i, j+1)
                r = return_value(img,i, j-1)
            elif (112.5 <= return_value(angle,i,j) < 157.5):
                q = return_value(img,i-1, j-1)
                r = return_value(img,i+1, j+1)
            elif (22.5 <= return_value(angle,i,j) < 67.5):
                q = return_value(img,i+1, j-1)
                r = return_value(img,i-1, j+1)
            p=90
            if (return_value(img,i,j) >= q) and (return_value(img,i,j) >= r):
                Z[i,j] = return_value(img,i,j)
                p=30
            else:
                Z[i,j] = 0

    img=Z


    cv2.imwrite('NMS.jpg',img)

    highThreshold=0.60
    lowThreshold=0.40

    highThreshold = img.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold

    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(75)
    strong = np.int32(255)

    zeros_i, zeros_j = np.where(img < lowThreshold)
    strong_i, strong_j = np.where(img >= highThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[weak_i, weak_j] = weak
    res[strong_i, strong_j] = strong

    res[zeros_i,zeros_j]=0

    final=res

    cv2.imwrite('output1.jpg',final) 

    n_white_pix = np.sum(img == 255)
    # print('Number of white pixels:', n_white_pix)

    density=n_white_pix/(M*N)
    print('Density:',density )

    if density>0.0005:
        result='Healthy eye'
    else:
        result='Hypertenstion detected'
    print(result)
    
    return result