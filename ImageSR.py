import torch
import sys
import os
import PIL.Image
import ImageDataset
import ImageSRModel
import torch.utils.data

train_dir = sys.argv[1]
target_dir = sys.argv[2]
train_dataset = os.listdir(train_dir)
target_dataset = os.listdir(target_dir)

lr_h = 510 
lr_w = 279


FACTOR = 4
TOTAL_ITERATIONS = 200000
BATCH_SIZE = 2
BATCHES = int(len(train_dataset) / BATCH_SIZE)
EPOCHS = int(TOTAL_ITERATIONS / len(train_dataset))

d_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir, lr_h, lr_w, FACTOR)
d_loaded = torch.utils.data.DataLoader(d_set, batch_size = BATCH_SIZE)

model_r = ImageSRModel.ImageSRModel(FACTOR)
optimizer_r = torch.optim.Adam(model_r.parameters())
model_r.train()
for epoch in range(EPOCHS):
    for img_name, lr_r_img, hr_r_img, lr_g_img, hr_g_img, lr_b_img, hr_b_img in d_loaded:
            
            pred = model_r(lr_r_img)
            #When doing the convolutions, you lose 1 pixel on both H and W, so when you upscale via PixelShuffle, you lose 4 on each spatial dimension
            #Thus, I cut 4 rows and 4 columns from target image. I suppose I could've used "crop()" instead
            desiredRows = torch.tensor(range((lr_h*FACTOR)-FACTOR))
            desiredCols = torch.tensor(range((lr_w*FACTOR)-FACTOR))
            hr_r_img = hr_r_img.view(-1,1,lr_h*FACTOR,lr_w*FACTOR)
        
            hr_r_img = torch.index_select(hr_r_img,2,desiredRows)
            hr_r_img = torch.index_select(hr_r_img,3,desiredCols)
            
            loss_r = torch.nn.functional.mse_loss(pred,hr_r_img)
            print("EPOCH: " + str(epoch) + " LOSS: " + str(loss_r))
            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()
     
