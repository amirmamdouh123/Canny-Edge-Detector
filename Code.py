import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt


def NonMaximalSuppression(img, radius):
 for y in range(len(img)-radius+1):
  for x in range(len(img[y])-radius+1):
      i_max = None
      j_max = None
      max = -1
      for i in range(radius):
          for j in range(radius):
            if img[y+i,x+j]>max:
                max = img[y+i,x+j]
                i_max= y+i
                j_max= x+j
            elif img[y+i,x+j]==max:
                img[y+i,x+j]= 0
            if i+1 == radius and j+1 == radius:
                img[i_max,j_max]= max
 return img

# 1- gradients in both the X and Y directions.
def harris(img, thresh=200, radius=2, verbose=True):
    Gx, Gy = utl.get_gradients_xy(img,5)
    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    # 2- smooth the derivative a little using gaussian
    #Student Code ~ 2 Lines
    Gx = cv2.GaussianBlur(Gx, (5, 5), sigmaX=3,sigmaY=0)
    Gy = cv2.GaussianBlur(Gy, (5, 5), sigmaX=0,sigmaY=3)
    #End Student Code
    
    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04



    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
    # 	3.2 Calculate M for each pixel:
    # 		    M = [[a11, a12],
    #                [a21, a22]]
    #           with a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2
            #Student Code ~ 1 line of code
            M = np.array([[int(Gx[i,j])*int(Gx[i,j]), int(Gx[i,j])*int(Gy[i, j])],
                          [int(Gx[i,j])*int(Gy[i,j]), int(Gy[i,j])*int(Gy[i, j])]])
            #Student Code

    # 	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
            Det_M = np.linalg.det(M)

    # 	3.4 Calculate Response at this pixel = det-k*trace^2
    #   where trace of M is the sum of its diagonals
            #Student Code ~ 1 line of code
            R[i, j] = Det_M - k*(M[0,0]+M[1, 1])**2

            #End Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255
    R[R<0]=0
    R = utl.rescale(R, 0, 255)

    # 5- Threshold and Non-Maximal Suppression
    # Student Code ~ 2 lines of code
    #R[R>thresh] = 255
    #R[R<=thresh] = 0
    #M = 0.01 * R.max()
    #M=-100
    # Threshold for an optimal value, it may vary depending on the image.

    R[R > 200] = 255
    R[R < 200] = 0
    plt.imshow(R, cmap="gray")
    plt.title("before Suppression")
    plt.show()
    Maxima = NonMaximalSuppression(R,20)

    # End Student Code
    plt.imshow(Maxima, cmap="gray")
    plt.title("after threshold")
    plt.show()

    return Maxima

img_pairs = [['check.bmp', 'check_rot.bmp']]
dir = 'input/'
i = 0;


for [img1,img2] in img_pairs:
    i += 1
    img1 = cv2.imread(dir+img1, 0)
    img2 = cv2.imread(dir+img2, 0)

    r1 = harris(img1)
    r2 = harris(img2) #Note that threshod may need to be different from picture to another
    plt.figure(i)
    plt.subplot(221), plt.imshow(img1, cmap='gray')
    plt.subplot(222), plt.imshow(img2, cmap='gray')
    plt.subplot(223), plt.imshow(r1, cmap='gray')
    plt.subplot(224), plt.imshow(r2, cmap='gray')
    plt.show()

