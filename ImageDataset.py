import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms.functional

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, train_image_names, train_dir, target_dir, crop_h, crop_w, channel, factor):
        self._train_names = train_image_names
        self._target_names = []
        self._train_dir = train_dir
        self._target_dir = target_dir
        self._cropped_h = crop_h
        self._cropped_w = crop_w
        self._factor = factor
        self._channel = channel
        for name in self._train_names:
            self._target_names.append(name.split("x")[0]+".png")

    def __len__(self):
        return len(self._train_names)

    def crop_image(self, image_lr, image_hr):
        return  \
        torchvision.transforms.functional.crop(image_lr,0,0,self._cropped_h,self._cropped_w), \
        torchvision.transforms.functional.crop(image_hr,0,0,self._cropped_h*self._factor,self._cropped_w*self._factor)
    
    def __getitem__(self, idx):
        train_img = PIL.Image.open(self._train_dir + "./" + self._train_names[idx])
        tar_img = PIL.Image.open(self._target_dir + "./" + self._target_names[idx])
        
        train_size = (train_img.size[1], train_img.size[0])
        tar_size = (tar_img.size[1], tar_img.size[0])
       
        train_tsr = torch.tensor(list(train_img.getdata(self._channel)), dtype=torch.float32).view(1, train_size[0],train_size[1])
        tar_tsr = torch.tensor(list(tar_img.getdata(self._channel)), dtype=torch.float32).view(1, tar_size[0],tar_size[1])

        train_img.close()
        tar_img.close()

        train_tsr, tar_tsr = self.crop_image(train_tsr,tar_tsr)

        #When doing the convolutions, you lose 1 pixel on both H and W, so when you upscale via PixelShuffle, you lose 4 on each spatial dimension
        #Thus, I cut 4 rows and 4 columns from target image. 
        #To be fair(?) I cut around the image, which felt more balanced than cutting just the farthest bottom and right edges
        desiredRows = torch.tensor(range((self._factor//2), (self._cropped_h*self._factor)-(self._factor//2)))
        desiredCols = torch.tensor(range((self._factor//2), (self._cropped_w*self._factor)-(self._factor//2)))
        
        tar_tsr = torch.index_select(tar_tsr,1,desiredRows)
        tar_tsr = torch.index_select(tar_tsr,2,desiredCols)
        
        print("Loaded: " + self._train_names[idx] ,self._target_names[idx] + " Sizes: " + str(train_tsr.shape), str(tar_tsr.shape))

        return \
        train_tsr,\
        tar_tsr
        #train_tsr.to('cuda')\
        #tar_tsr.to('cuda')
        
        
