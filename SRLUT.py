import torch
import sys
import os
import math
import PIL.Image
import ImageSRModel
import ImageDataset
import torch.utils.data
import torchvision
import numpy
import json

#These are the hard sizes that the LR images are cut to
lr_h = 510 
lr_w = 279
#The SRLUT class has functions to build the table and test it, as well as some internal helpers
class SRLUT:
    #Constructor, take the parameters that define how the table is built
    #Unfortunately, this actually doesn't generalize well to different sizes of CONV_KERNEL, because I've assumed a square kernel
    #The paper doesn't, but I don't know how to deal with a non-square kernel for any of the rest of the model, so I fixed that assumption
    #RF is Receptive Field size, CONV_KERNEL is a dimension of the assumed SQUARE convolution kernel
    #FACTOR is the factor by which the LR images were downscaled, and SAMP_INTV is the sampling interval of the SRLUT
    def __init__(self, RF, CONV_KERNEL, FACTOR, SAMP_INTV):
        self._RF = RF
        self._KERNEL = CONV_KERNEL
        self._FACTOR = FACTOR
        self._SAMP_INTV = SAMP_INTV

    #A helper that retrieves a square of pixels, if size is 1, Give me a 1x1 square, if size is 2, Give me a 2x2, etc.
    #h_loc and w_loc are the top left corner of the desired square
    #max_h and max_w are the actual size of the image
    def _get_img_square(self, img, size, h_loc, w_loc, max_h, max_w):
        square = []
        for h in range(size):
            row_pix = []
            for w in range(size):
                #For every pixel relative to the top left corner we start at, we check if the neighbour pixels either right or below us 
                #are out of the range of the image, if so, we assume a 0 for those out of bound pixels
                if (h + h_loc >= max_h or w + w_loc >= max_w):
                    row_pix.append(0)
                #If the pixel is in bounds, then give the colour at that pixel, put it in a "row list", which has pixels along this row
                else:
                    row_pix.append(int(img[h_loc+h][w_loc+w]))
            #We finish a row, put that row into the square list, which is essentially a stack of rows, like a square
            square.append(row_pix)
        return torch.tensor(square).view(size,size)

    def _build_table(self, model):
        #Size of table will be (2^8)^RF x FACTOR^2 x 8bits, which would be huge, so they sample it uniformly.
        #They sample every 2^4 values in the table, and interpolate between them.
        sr_lut = {}
        #First we build images that will have 2x2 segments that we actually want in our table (Pixel Colours at multiples of 16, or 255 exactly)
        #We build 4 images, both in the LR scale, so max height and width of lr_h and lr_w
        images = []
        for i in range(10):
            #Initial pixel values are all -1 (not an actual colour)
            img = torch.zeros((1, 1, lr_h, lr_w)) - 1
            rand = numpy.random.default_rng()
            for h in range(lr_h):
                for w in range(lr_w):
                    n = rand.integers(0,17)
                    if n == 16:
                        img[0][0][h][w] = 255
                    else:
                        img[0][0][h][w] = n*self._SAMP_INTV
            images.append(img)

        for lr_img in images:
            pred_hr_sq = model(lr_img)
            #Knock out the batch and channel dimensions - only the model needs those
            lr_img = lr_img[0][0]
            pred_hr_sq = pred_hr_sq[0][0]
            #Interesting consequence, the model predictions are not exactly x4 scaled up from the raw image
            #The prediction, due to the initial 2x2 convolution, loses one row and one column, so on the final DTS transformation,
            #it is 4 rows and 4 columns shorter than the actual HR image. To make things fit, I have to cut the incoming LR image here
            #Since it doesn't align perfectly with the x4 scale up.
            lr_img = torchvision.transforms.functional.crop(lr_img, 1, 1, lr_img.shape[0]-2, lr_img.shape[1]-2)
            max_h = lr_img.shape[0]
            max_w = lr_img.shape[1]
            for ind_h in range(max_h):
                for ind_w in range(max_w):
                    lr_sq = self._get_img_square(lr_img, int(self._RF/self._KERNEL), ind_h, ind_w, max_h, max_w)
                    valid = True
                    #Check that all the colours are actually multiples of 16 (which is how we're sampling) or exactly 255
                    for i in range(lr_sq.shape[0]):
                        for j in range (lr_sq.shape[1]):
                            if (lr_sq[i][j] % self._SAMP_INTV == 0 or lr_sq[i][j] == 255):
                                valid = valid & True
                            else:
                                valid = valid & False
                            if valid == False:
                                break
                    if (valid):
                        hr_sq = self._get_img_square(pred_hr_sq, self._FACTOR, self._FACTOR*ind_h, self._FACTOR*ind_w, self._FACTOR*max_h, self._FACTOR*max_w)
                        key = tuple((lr_sq.flatten()).tolist())
                        if key not in sr_lut:
                            sr_lut[key] = tuple((hr_sq.flatten()).tolist())
                            #print(len(sr_lut.keys()))
                        else:
                            max_ind = (self._SAMP_INTV) + 1 #Maximum number of elements at each index due to only inputting every 2^4th colour as a key element (or 255)
                            if (len(sr_lut.keys()) == max_ind**4):
                                #If we already have all the possible keys in our sampled table, stop doing work, and return the table
                                print("TABLE FULL")
                                return sr_lut
        print("UNDERFULL")
        return sr_lut

    def _P(self, Point_Index, MSB_key, c_table):
        lookup_index = []
        for i in range(len(Point_Index)):
            table_ind = (self._SAMP_INTV*(MSB_key[i]+Point_Index[i]))
            if table_ind > 255:
                table_ind = 255
            lookup_index.append(table_ind)
        return torch.tensor(c_table[tuple(lookup_index)])

    def _tetra_interpol(self, W, MSB_key, LSB_key, c_table):
        #The MSBs of the query colour I0 I1 I2 I3, in that order, define the fixed points
        #The LSBs will define the weights assigned to the interpolation
        O = []
        #Add the first and last bounding vertexes (P_0000) and (P_1111)
        O.append(self._P([0,0,0,0], MSB_key, c_table))
        O.append(self._P([1,1,1,1], MSB_key, c_table))
        Point_Indexer = [0,0,0,0]
        #We'll use this to conveniently find the weights of the interpolation
        LSB_order = [W,0]
        LSB_tsr = torch.tensor([LSB_key[0], LSB_key[1], LSB_key[2], LSB_key[3]])
        for i in range(3):
            largest = torch.argmax(LSB_tsr)
            #Conveniently, the argmax torch gives is the same index for the LSB_key that is the largest
            LSB_order.insert(i+1, LSB_key[largest])
            #Take it out of the running
            LSB_tsr[largest] = -1
            #Define the (i+1)th bounding vertex
            Point_Indexer[largest] = 1
            #Put the bounding vertex into the list as the (i+1)th bounding vertex
            O.insert((i+1),self._P(Point_Indexer, MSB_key, c_table))
        #Now we have the bounding vertices calculated, from O0 to O4
        #We just need the weights
        weights=[]
        for i in range(len(LSB_order)-1):
            weights.append(LSB_order[i] - LSB_order[i+1])
        interpolation = 0
        for i in range(len(weights)):
            interpolation += weights[i] * O[i]
        interpolation = (1/W) * interpolation
        return interpolation.view(self._FACTOR, self._FACTOR)

    
    def test_table(self, lr_input, c_table):
        max_h = len(lr_input)
        max_w = len(lr_input[0])
        row_tensors = []
        for h in range(max_h):
            col_tensors = []
            for w in range(max_w):
                lr_sq = self._get_img_square(lr_input, 2, h, w, max_h, max_w)
                key = (lr_sq.flatten()).tolist()
                #If the key is actually in the table, we're good to go, just fetch it, shape it to the corresponding shape, and return it
                if tuple(key) in c_table.keys():
                    col_tensors.append(torch.tensor(c_table[tuple(key)]).view(self._FACTOR,self._FACTOR))
                else:
                    #The values in the key will not be perfect multiples of 16 this time, so we need to:
                    #First, find the nearest entry in the table, Second, interpolate between entries to get the value for this input
                    #According to the paper, take the elements of the key, and get the 4 MSB and 4 LSB of each element
                    MSB_key = key
                    LSB_key = key
                    #The keys are in order, from the top left pixel to bottom left pixel of the 2x2 LR blob
                    #Following notation of the paper, that's I0, I1, I2, I3
                    for i in range(len(key)):
                        MSB_key[i] = (MSB_key[i] & 240) >> 4
                        LSB_key[i] = (LSB_key[i] & 15)
                    #We use the MSBs to define fixed points, and the LSBs to determine weights
                    #Following the table provided by the paper
                    col_tensors.append(self._tetra_interpol(2**4, tuple(MSB_key), tuple(LSB_key), c_table))
            row_tensors.append(torch.hstack(col_tensors))
        return torch.vstack(row_tensors).tolist()

    def produce_tables(self, model, col):
        j = json.JSONEncoder()
        model.eval()
        t = self._build_table(model)
        to_json = {}
        for keys in t.keys():
            to_json[str(keys)] = t[keys]
        f = open(col+"_table.json", "w")
        for segment in j.iterencode(to_json):
            f.write(segment)
        f.close()
    
    def decode_table(self, fp_json_table):
        table = fp_json_table.readline()
        j = json.JSONDecoder()
        t = j.decode(table)
        from_json = {}
        for keys in t.keys():
            tuple_elem_list = keys.strip("()").split(", ")
            for i in range(len(tuple_elem_list)):
                tuple_elem_list[i] = int(tuple_elem_list[i])
            from_json[tuple(tuple_elem_list)] = t[keys]
        del t
        return from_json
