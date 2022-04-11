import PIL.Image
import torch
import torch.utils.data
import torchvision.transforms.functional

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, train_image_names, train_dir, target_dir, crop_h, crop_w, factor):
        self._train_names = train_image_names
        self._target_names = []
        self._train_dir = train_dir
        self._target_dir = target_dir
        self._cropped_h = crop_h
        self._cropped_w = crop_w
        self._factor = factor
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
       
        train_r_tsr = torch.tensor(list(train_img.getdata(0)), dtype=torch.float32).view(1, train_size[0],train_size[1])
        tar_r_tsr = torch.tensor(list(tar_img.getdata(0)), dtype=torch.float32).view(1, tar_size[0],tar_size[1])

        train_g_tsr = torch.tensor(list(train_img.getdata(1)), dtype=torch.float32).view(1, train_size[0],train_size[1])
        tar_g_tsr = torch.tensor(list(tar_img.getdata(1)), dtype=torch.float32).view(1, tar_size[0],tar_size[1])

        train_b_tsr = torch.tensor(list(train_img.getdata(2)), dtype=torch.float32).view(1, train_size[0],train_size[1])
        tar_b_tsr = torch.tensor(list(tar_img.getdata(2)), dtype=torch.float32).view(1, tar_size[0],tar_size[1])

        train_img.close()
        tar_img.close()

        train_r_tsr, tar_r_tsr = self.crop_image(train_r_tsr,tar_r_tsr)
        train_g_tsr, tar_g_tsr = self.crop_image(train_g_tsr,tar_g_tsr)
        train_b_tsr, tar_b_tsr = self.crop_image(train_b_tsr,tar_b_tsr)

        print("Loaded: " + self._train_names[idx] ,self._target_names[idx] + " Sizes: " + str(train_r_tsr.shape), str(tar_r_tsr.shape))

        return self._train_names[idx], train_r_tsr, tar_r_tsr, train_g_tsr, tar_g_tsr, train_b_tsr, tar_b_tsr
        
        
