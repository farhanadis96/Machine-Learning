import numpy as np
import cv2

class convolution:

    def __init__(self,fanin,fanout):
        self.bias=np.zeros((1,1,1,64))
        self.filter=np.random.randn(3,3,fanin,fanout)*np.sqrt(2/fanin)*0.01
        #return self.filter,self.bias
    def conv_forward(self,inputimage, filter, stride, pad, bias):

        (batchsize, inp_height, inp_width, inp_chan) = inputimage.shape
        (n_height, n_width, n_channels, n_filters) = filter.shape

        future_h = ((inp_height - n_height + 2 * pad) / stride) + 1
        future_w = ((inp_width - n_width + 2 * pad) / stride) + 1

        fut_img = np.zeros((batchsize, int(future_h), int(future_w), n_filters))

        inputimage = np.pad(inputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

        for i in range(batchsize):
            inputimage = inputimage[i]
            for k in range(inp_height):
                for m in range(inp_width):
                    for c in range(n_filters):
                        h_init = k * stride
                        h_end = h_init + 3
                        w_init = m * stride
                        w_end = w_init + 3
                        inputpart = inputimage[h_init:h_end, w_init:w_end, :]
                        fut_img[i, k, m, c] = self.convinput(inputpart, filter[:, :, :, c], bias[:, :, :, c])
        cache = (inputimage, filter, stride, pad, bias)
        return fut_img, cache

    def convinput(self,input, filter, bias):
        t = filter * input + bias
        return np.sum(t)
    def conv_backward(self,dZ, cache):



        # Retrieve information from "cache"
        (inputimage, filter, stride, pad, bias) = cache

        # Retrieve dimensions from A_prev's shape
        (batchsize, inp_height, inp_width, inp_chan) = inputimage.shape

        # Retrieve dimensions from W's shape
        (n_height, n_width, n_channels, n_filters) = filter.shape

        # Retrieve dimensions from dZ's shape
        (batchsize, height, width, chan) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dinputimage = np.zeros((batchsize, inp_height, inp_width, inp_chan))
        dinput= np.zeros((batchsize, inp_height, inp_width, inp_chan))
        dWeight = np.zeros((3, 3, inp_chan, chan))
        dbias = np.zeros((1, 1, 1, chan))

        # Pad A_prev and dA_prev
        inputimage = np.pad(inputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        dinputimage = np.pad(dinputimage, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

        for i in range(batchsize):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            inputimage = inputimage[i]
            dinputimage = dinputimage[i]

            for h in range(height):  # loop over vertical axis of the output volume
                for w in range(width):  # loop over horizontal axis of the output volume
                    for c in range(chan):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * stride

                        vert_end = vert_start + n_height
                        horiz_start = w * stride

                        horiz_end = horiz_start + n_width

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = inputimage[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        dinputimage[vert_start:vert_end, horiz_start:horiz_end, :] += filter[:, :, :, c] * dZ[
                            i, h, w, c]
                        dWeight[:, :, :, c] += a_slice * dZ[i, h, w, c]
                        dbias[:, :, :, c] += dZ[i, h, w, c]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dinput[i,:, :, :] = dinputimage[pad:-pad, pad:-pad, :]
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (dinput.shape == (batchsize, inp_height, inp_width, inp_chan))

        return dinput, dWeight, dbias