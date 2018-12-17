

class relu:

    def forward(self,image):
        image[image<0]=0
        return image

    def relu_backward (self,image):
        image[image<0]=0
        image[image>=0]=1
        return image