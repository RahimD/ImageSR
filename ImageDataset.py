import PIL.Image
import torch
import torch.utils.data

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, train_image_names, train_dir, target_dir):
        self._train_names = train_image_names
        self._target_names = []
        self._train_dir = train_dir
        self._target_dir = target_dir
        for name in self._train_names:
            self._target_names.append(name.split("x")[0]+".png")

    def __len__(self):
        return len(self._train_names)

    def __getitem__(self, idx):
        train_img = PIL.Image.open(self._train_dir + "./" + self._train_names[idx])
        tar_img = PIL.Image.open(self._target_dir + "./" + self._target_names[idx])
        
        train_size = (train_img.size[1], train_img.size[0])
        tar_size = (tar_img.size[1], tar_img.size[0])
       
        train_r_tsr = torch.tensor(list(train_img.getdata(0)), dtype=torch.float32)
        tar_r_tsr = torch.tensor(list(tar_img.getdata(0)), dtype=torch.float32)

        train_g_tsr = torch.tensor(list(train_img.getdata(1)), dtype=torch.float32)
        tar_g_tsr = torch.tensor(list(tar_img.getdata(1)), dtype=torch.float32)

        train_b_tsr = torch.tensor(list(train_img.getdata(2)), dtype=torch.float32)
        tar_b_tsr = torch.tensor(list(tar_img.getdata(2)), dtype=torch.float32)

        train_img.close()
        tar_img.close()


        print("Loaded: " + self._train_names[idx] ,self._target_names[idx] + " Sizes: " + str(train_r_tsr.shape), str(tar_r_tsr.shape))

        return self._train_names[idx], train_r_tsr, tar_r_tsr, train_g_tsr, tar_g_tsr, train_b_tsr, tar_b_tsr, train_size, tar_size
        
        