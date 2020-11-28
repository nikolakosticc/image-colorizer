import numpy as np 
import cv2

print("loading models.....")
net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt.txt','colorization_release_v2.caffemodel')
pts = np.load('pts_in_hull.npy')


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]


image = cv2.imread('albert_einstein.jpg') #Adding the original image
scaled = image.astype("float32")/255.0 #Scaling pixel intensities to the range [0, 1]   (RGB)
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB) #Converting from BGR to Lab color space   (BGR is reversed RGB color space)


resized = cv2.resize(lab,(224,224)) #Resize the input image to 224px×224px, the required input dimensions for the network
L = cv2.split(resized)[0]
L -= 50


net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (image.shape[1],image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)

colorized = (255 * colorized).astype("uint8")

cv2.imshow("Original",image) #Shows original image
cv2.imshow("Colorized",colorized) #Shows colorized image

cv2.waitKey(0) #displays the window infinitely until any keypress