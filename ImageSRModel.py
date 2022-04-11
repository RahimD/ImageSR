import torch.nn

class ImageSRModel(torch.nn.Module):
    def __init__(self, FACTOR):
        super().__init__()
        self._L1 =  torch.nn.Conv2d(1,64, kernel_size=(2,2))               #L1
        self._L2 =  torch.nn.Conv2d(64,64, kernel_size=(1,1))              #L2
        self._L3 =  torch.nn.Conv2d(64,64, kernel_size=(1,1))              #L3
        self._L4 =  torch.nn.Conv2d(64,64, kernel_size=(1,1))              #L4
        self._L5 =  torch.nn.Conv2d(64,64, kernel_size=(1,1))              #L5
        self._L6 =  torch.nn.Conv2d(64,FACTOR*FACTOR, kernel_size=(1,1))   #L6
        self._DTS =  torch.nn.PixelShuffle(FACTOR)                         #Depth-To-Space

    def forward(self, input_batch):
        rot_ens = []
        for i in range(4):
            net_in = torch.rot90(input_batch,i,[2,3])
            
            net_in = torch.nn.functional.relu(self._L1(net_in))
            net_in = torch.nn.functional.relu(self._L2(net_in))
            net_in = torch.nn.functional.relu(self._L3(net_in))
            net_in = torch.nn.functional.relu(self._L4(net_in))
            net_in = torch.nn.functional.relu(self._L5(net_in))
            net_in = self._L6(net_in)
            net_in = self._DTS(net_in)

            net_out = torch.rot90(net_in,-i,[2,3])
            rot_ens.append(net_out)
            print("ROTATION: " + str(i*90))
        final_pred = rot_ens[0] + rot_ens[1] + rot_ens[2] + rot_ens[3]
        final_pred = 0.25 * final_pred
        return final_pred



        