import torch
import sys
import os
import math
import PIL.Image
import ImageSRModel
import ImageDataset
import torch.utils.data
import torchvision

train_dir = sys.argv[1]
target_dir = sys.argv[2]
train_dataset = os.listdir(train_dir)
target_dataset = os.listdir(target_dir)

lr_h = 510 
lr_w = 279

class SRLUT:
    def __init__(self, RF, CONV_KERNEL, FACTOR, SAMP_INTV, d_set, model):
        self._RF = RF
        self._KERNEL = CONV_KERNEL
        self._FACTOR = FACTOR
        self._SAMP_INTV = SAMP_INTV
        self._build_table(d_set, model)
    #size is 1, Give me a 1x1, size is 2, Give me a 2x2, etc.
    #h_loc and w_loc are the top left corner of the desired square
    def _get_img_square(self, img, size, h_loc, w_loc, max_h, max_w):
        square = []
        for h in range(size):
            row_pix = []
            for w in range(size):
                if (h + h_loc >= max_h or w + w_loc >= max_w):
                    row_pix.append(0)
                else:
                    row_pix.append(int(img[h_loc+h][w_loc+w]))
            square.append(row_pix)
        return torch.tensor(square)

    def _build_table(self, d_set, model):
        #Size of table will be (2^8)^RF x FACTOR^2 x 8bits, which would be huge, so they sample it uniformly.
        #They sample every 2^4 values in the table, and interpolate between them.
        sr_lut = {}
        d_loaded = torch.utils.data.DataLoader(d_set, batch_size = 1)
        for lr_img, hr_img in d_loaded:
            #Knock out the batch and channel dimensions, there's only one image in the batch, and it's a single channel
            lr_img = lr_img[0,0]
            hr_img = hr_img[0,0]
            
            #Interesting consequence, the model predictions are not exactly x4 scaled up from the raw image
            #The prediction, due to the initial 2x2 convolution, loses one row and one column, so on the final DTS transformation,
            #it is 4 rows and 4 columns shorter than the actual HR image. To make things fit, I have to cut the incoming LR image here
            #Since it doesn't align perfectly with the x4 scale up.
            #pred_hr_sq = model(lr_img)
            lr_img = torchvision.transforms.functional.crop(lr_img, 1, 1, lr_img.shape[0]-2, lr_img.shape[1]-2)
            max_h = lr_img.shape[0]
            max_w = lr_img.shape[1]
            for ind_h in range(max_h):
                for ind_w in range(max_w):
                    lr_sq = self._get_img_square(lr_img, int(self._RF/self._KERNEL), ind_h, ind_w, max_h, max_w)
                    valid = True
                    #Check that all the colours are actually multiples of 16 (which is how we're sampling) or exactly 255
                    for i in range(len(lr_sq)):
                        if (lr_sq[i] % self._SAMP_INTV == 0 or lr_sq[i] == 255):
                            valid = valid & True
                        if valid == False:
                            break
                    if (valid):
                        hr_sq = self._get_img_square(hr_img, self._FACTOR, self._FACTOR*ind_h, self._FACTOR*ind_w, self._FACTOR*max_h, self._FACTOR*max_w, False)
                        #hr_sq = get_img_square(pred_hr_img, self._FACTOR, self._FACTOR*ind_h, self._FACTOR*ind_w, self._FACTOR*max_h, self._FACTOR*max_w, False)
                        key = tuple((lr_sq.flatten()).tolist())
                        if key not in sr_lut:
                            sr_lut[key] = tuple((hr_sq.flatten()).tolist())
                            print(len(sr_lut.keys()))
                        else:
                            max_ind = (256/(self._SAMP_INTV)) + 1 #Maximum number of elements at each index due to only inputting every 2^4th colour as a key element (or 255)
                            if (len(sr_lut.keys()) == max_ind**4):
                                #If we already have all the possible keys in our sampled table, stop doing work, and return the table
                                return sr_lut
        return sr_lut

    def P(Point_Index, MSB_key, c_table):
        lookup_index = []
        for i in range(len(Point_Index)):
            lookup_index[i] = (self._SAMP_INTV*(MSB_key[i]+Point_Index[i]))
            if lookup_index[i] > 255:
                lookup_index[i] = 255
        return torch.tensor(c_table[(lookup_index[0], lookup_index[1], lookup_index[2], lookup_index[3])])

    def tetra_interpol(W, MSB_key, LSB_key, c_table):
        #The MSBs of the query colour I0 I1 I2 I3, in that order, define the fixed points
        #The LSBs will define the weights assigned to the interpolation
        O = []
        #Add the first and last bounding vertexes (P_0000) and (P_1111)
        O.append(P([0,0,0,0] MSB_key, c_table))
        O.append(P([1,1,1,1] MSB_key, c_table))
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
            O.insert((i+1),P(Point_Indexer, MSB_key, c_table))
        #Now we have the bounding vertices calculated, from O0 to O4
        #We just need the weights
        weights=[]
        for i in range(len(LSB_order)-1):
            weights.append(LSB_order[i] - LSB_order[i+1])
        interpolation = 0
        for i in range(len(weights)):
            interpolation += weights[i] * O[i]
        interpolation = (1/W) * interpolation
        return interpolation.view(self._KERNEL*self._FACTOR, self._KERNEL*self._FACTOR)

    
    def test_table(lr_input, c_table):
        max_h = lr_input.shape[0]
        max_w = lr_input.shape[1]
        row_tensors = []
        for h in range(max_h):
            col_tensors = []
            for w in range(max_w):
                lr_sq = self._get_img_square(self, lr_input, 2, h, w, max_h, max_w):
                key = tuple((lr_sq.flatten()).tolist())
                #If the key is actually in the table, we're good to go, just fetch it, shape it to the corresponding shape, and return it
                if key in c_table.keys():
                    col_tensors.append(torch.row_tensor,torch.tensor(c_table[key]).view(2*self._FACTOR,2*self._FACTOR))
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
                    col_tensors.append(self._tetra_interpol(2**4, MSB_key,LSB_key, c_table))
            row_tensors.append(torch.hstack(col_tensors))
        return torch.vstack(row_tensors)
                


sr_lut = []
for channel in range(3):
    model = torch.load("model_" + c[channel] + ".pth")
    model.eval()
    d_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir, lr_h, lr_w, channel, FACTOR)
    table = SRLUT(RF,CONV_KERNEL,FACTOR, d_set, model)
    sr_lut.append(table)
