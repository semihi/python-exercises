import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


#------------!Çerçeve Boyutlama!---------------
def rescaleFrame(frame, scale): #sahnemizi boyutlandırmak için bir fonksyon tanımladık ve 2 değişken alıyor sahne ve boyut
    width = int(frame.shape[1] * scale) #shape[1] sahnenin enini döndürürken
    height = int(frame.shape[0] * scale) #shape[0] sahnenin yükesliğini döndürür
    #isterseniz 0 ve 1 in yerini değiştirip olacakları deneyin
    dimensions = (width,height) #ebatlarımızı bir değişkene atadık
    return cv.resize(frame, dimensions) #ardından ebatlarımızı sahnenin üstünde uygulayıp geri döndürdük

class DepthMap:
    def __init__(self, showImages):
        # Load Images
        root = os.getcwd()
        imgLeftPath = 'orange_left.jpg'
        imgRightPath = 'orange_right.jpg'
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)
        #self.imgLeft = rescaleFrame(cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE), 0.1)
        #self.imgRight = rescaleFrame(cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE), 0.1)

        # Kırpılacak alanın koordinatlarını belirleyin
        startX = 5  # Sol üst köşe X koordinatı
        startY = 1226  # Sol üst köşe Y koordinatı
        width = 2265  # Kırpılacak bölgenin genişliği
        height = 729  # Kırpılacak bölgenin yüksekliği

        # Resimlerin her ikisini de aynı şekilde kırpın
        self.imgLeft_cropped = self.imgLeft[startY:startY + height, startX:startX + width]
        self.imgRight_cropped = self.imgRight[startY:startY + height, startX:startX + width]


        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft_cropped)
            plt.subplot(122)
            plt.imshow(self.imgRight_cropped)
            # plt.show()

    def computeDepthMapBM(self):  # Block Matching
        nDispFactor = 24  # adjust this
        stereo = cv.StereoBM.create(numDisparities=16 * nDispFactor,
                                    blockSize=5)
        disparity = stereo.compute(self.imgLeft_cropped, self.imgRight_cropped)
        plt.figure()
        plt.imshow(disparity, 'gray')
        plt.title('BM')
        plt.show()

    def computeDepthMapSGBM(self):
        # SGBM:
        window_size = 7
        min_disp = 16
        nDispFactor = 14  # adjust this (14 is good)
        num_disp = 7 * nDispFactor - min_disp

        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=window_size,
                                      P1=8 * 3 * window_size ** 2,
                                      P2=32 * 3 * window_size ** 2,
                                      disp12MaxDiff=1,
                                      uniquenessRatio=15,
                                      speckleWindowSize=0,
                                      speckleRange=2,
                                      preFilterCap=63,
                                      mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

        # Compute disparity map
        disparity = stereo.compute(self.imgLeft_cropped, self.imgRight_cropped).astype(np.float64) / 16.0

        # Display the disparity map
        plt.figure()
        plt.imshow(disparity, 'gray')
        plt.title('SGBM')
        plt.colorbar()
        plt.show()

def demoViewPics():
        # See pictures
        dp = DepthMap(showImages=True)

def demoStereoBM():
        dp = DepthMap(showImages=False)
        dp.computeDepthMapBM()

def demoStereoSGBM():
        dp = DepthMap(showImages=False)
        dp.computeDepthMapSGBM()

if __name__ == '__main__':
    demoViewPics()
    # demoStereoBM()
    demoStereoSGBM()