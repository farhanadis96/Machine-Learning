from convolution import convolution
from relu import relu
import os
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error


def l2loss(x,y):
    loss=((x - y)**2).mean(axis=None)
    gradient = y-x
    return loss,gradient


def update_parameters(filter,gradient, bias, biasgradient):
    alpha=0.001
    filter= filter - (alpha*gradient)
    bias=bias-(alpha*biasgradient)
    return filter,bias

l1=convolution(3,64)
filter1=l1.filter
bias1=l1.bias

print(filter1,bias1)

l2=convolution(64,64)
filter2=l2.filter
bias2=l2.bias



l3=convolution(64,64)
filter3=l3.filter
bias3=l3.bias



l4=convolution(64,64)
filter4=l4.filter
bias4=l4.bias


l5=convolution(64,64)
filter5=l5.filter
bias5=l5.bias

l6=convolution(64,64)
filter6=l6.filter
bias6=l6.bias


l7=convolution(64,64)
filter7=l7.filter
bias7=l7.bias

l8=convolution(64,3)
filter8=l8.filter
bias8=l8.bias
imageBlur=[]
imageSharp=[]
for subdir, dirs, files in os.walk("./"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg") and ("Blur" in filepath):
            imageBlur.append(filepath)
        if filepath.endswith(".jpg") and ("sharp" in filepath):
            imageSharp.append(filepath)


for imag,imagSharp in zip(imageBlur,imageSharp):

    # forward propagation
    conv=convolution(3,3)
    reluo=relu()
    image = cv2.imread(imag)
    image=np.array(image)
    image=image[np.newaxis,...]
    print(image.shape)
    imageSharp=cv2.imread(imagSharp)
    imageSharp=np.array(imageSharp)



    image,cache1=conv.conv_forward(image,filter1,1,1,bias1)
    image=reluo.forward(image)

    image, cache2 = conv.conv_forward(image, filter2, 1, 1, bias2)
    image = reluo.forward(image)

    image, cache3 = conv.conv_forward(image, filter3, 1, 1, bias3)
    image = reluo.forward(image)

    image, cache4 = conv.conv_forward(image, filter4, 1, 1, bias4)
    image = reluo.forward(image)

    image, cache5 = conv.conv_forward(image, filter5, 1, 1, bias5)
    image = reluo.forward(image)

    image, cache6 = conv.conv_forward(image, filter6, 1, 1, bias6)
    image = reluo.forward(image)

    image, cache7 = conv.conv_forward(image, filter7, 1, 1, bias7)
    image = reluo.forward(image)

    image, cache8 = conv.conv_forward(image, filter8, 1, 1, bias8)
    (inputimage, filter, stride, pad, bias) = cache8
    inputimage = inputimage[np.newaxis, ...]
    cache8=(inputimage,filter,stride,pad,bias)
    # loss function
    loss,gradient = l2loss(image, imageSharp)
    print(loss)
    # backward propagation
    # pararmeters updation
    dimage,dW,Db=conv.conv_backward(gradient,cache8)
    filter8,bias8=update_parameters(filter8,dW,bias8,Db)
    dimage=reluo.relu_backward(dimage)

    dimage, dW, Db = conv.conv_backward(dimage, cache7)
    filter7, bias7 = update_parameters(filter7, dW, bias7, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = conv.conv_backward(dimage, cache6)
    filter6, bias6 = update_parameters(filter6, dW, bias6, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = conv.conv_backward(dimage, cache5)
    filter5, bias5 = update_parameters(filter5, dW, bias5, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = conv.conv_backward(dimage, cache4)
    filter4, bias4 = update_parameters(filter4, dW, bias4, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = convolution.conv_backward(dimage, cache3)
    filter3, bias3 = update_parameters(filter3, dW, bias3, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = convolution.conv_backward(dimage, cache2)
    filter2, bias2 = update_parameters(filter2, dW, bias2, Db)
    dimage = reluo.relu_backward(dimage)

    dimage, dW, Db = convolution.conv_backward(dimage, cache1)
    filter1, bias1 = update_parameters(filter1, dW, bias1, Db)


# testing
testInput=[]
for subdir, dirs, files in os.walk("./result/input"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg") or filepath.endswith(".png"):
            testInput.append(filepath)
for imag in testInput:
    count=1
    image = cv2.imread(imag)
    image = np.array(image)
    image = image[np.newaxis, ...]
    #200*200   to 500*500

    for i in range(150):
        image, cache1 = conv.conv_forward(image, filter1, 1, 1, bias1)
        image = reluo.forward(image)

        image, cache2 = conv.conv_forward(image, filter2, 1, 1, bias2)
        image = reluo.forward(image)

        image, cache3 = conv.conv_forward(image, filter3, 1, 1, bias3)
        image = reluo.forward(image)

        image, cache4 = conv.conv_forward(image, filter4, 1, 1, bias4)
        image = reluo.forward(image)

        image, cache5 = conv.conv_forward(image, filter5, 1, 1, bias5)
        image = reluo.forward(image)

        image, cache6 = conv.conv_forward(image, filter6, 1, 1, bias6)
        image = reluo.forward(image)

        image, cache7 = conv.conv_forward(image, filter7, 1, 1, bias7)
        image = reluo.forward(image)

        image, cache8 = conv.conv_forward(image, filter8, 1, 1, bias8)

        image = cv2.resize(imag, None, fx=1.004016, fy=1.004016, interpolation=cv2.INTER_CUBIC)
        image = image[np.newaxis, ...]

    imagname=filepath.replace(imag,"input","output")
    cv2.imwrite(imagname,image)




