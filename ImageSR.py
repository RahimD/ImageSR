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

FACTOR = 4
TOTAL_ITERATIONS = 200000
BATCH_SIZE = 32
BATCHES = int(len(train_dataset) / BATCH_SIZE)
EPOCHS = int(TOTAL_ITERATIONS / len(train_dataset))

d_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir)
model_r = ImageSRModel.ImageSRModel(FACTOR)
optimizer_r = torch.optim.Adam(model_r.parameters())
model_r.train()
for epoch in range(EPOCHS):
    for batch_num in range(BATCHES):
        for datum in range(batch_num*BATCH_SIZE,(batch_num+1)*BATCH_SIZE):
            img_name, lr_r_img, hr_r_img, lr_g_img, hr_g_img, lr_b_img, hr_b_img, lr_size, hr_size = d_set[datum]
            print(lr_size)
            
            pred = model_r(lr_r_img, lr_size)
            desiredRows = torch.tensor(range(hr_size[0]-4))
            desiredCols = torch.tensor(range(hr_size[1]-4))
            hr_r_img = hr_r_img.view(-1,1,hr_size[0],hr_size[1])
        
            hr_r_img = torch.index_select(hr_r_img,2,desiredRows)
            hr_r_img = torch.index_select(hr_r_img,3,desiredCols)
            
            loss_r = torch.nn.functional.mse_loss(pred,hr_r_img)
            print("EPOCH: " + str(epoch) + " LOSS: " + str(loss_r))
            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()
     
