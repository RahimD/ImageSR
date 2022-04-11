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
TOTAL_ITERATIONS = 800
BATCH_SIZE = 1
BATCHES = int(len(train_dataset) / BATCH_SIZE)
EPOCHS = int(TOTAL_ITERATIONS / len(train_dataset))

d_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir, lr_h, lr_w, FACTOR)
d_loaded = torch.utils.data.DataLoader(d_set, batch_size = BATCH_SIZE, shuffle = True)

v_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir, lr_h, lr_w, FACTOR)
v_loaded = torch.utils.data.DataLoader(v_set, batch_size = 2*BATCH_SIZE)

model_r = ImageSRModel.ImageSRModel(FACTOR)
model_b = ImageSRModel.ImageSRModel(FACTOR)
model_g = ImageSRModel.ImageSRModel(FACTOR)
optimizer_r = torch.optim.Adam(model_r.parameters())
optimizer_g = torch.optim.Adam(model_g.parameters())
optimizer_b = torch.optim.Adam(model_b.parameters())
for epoch in range(EPOCHS):
    batch_num = 0
    batch_losses = [[],[],[]]
    model_r.train()
    model_g.train()
    model_b.train()
    for img_name, lr_r_img, hr_r_img, lr_g_img, hr_g_img, lr_b_img, hr_b_img in d_loaded:
            print("EPOCH: " + str(epoch) + "BATCH: " + str(batch_num))
            pred_r = model_r(lr_r_img)
            pred_g = model_g(lr_g_img)
            pred_b = model_b(lr_b_img)
            
            loss_r = torch.nn.functional.mse_loss(pred_r,hr_r_img, reduction="sum")
            loss_g = torch.nn.functional.mse_loss(pred_g,hr_g_img, reduction="sum")
            loss_b = torch.nn.functional.mse_loss(pred_b,hr_b_img, reduction="sum")
            
            print(float(loss_r))
            print(float(loss_g))
            print(float(loss_b))

            batch_losses[0].append(float(loss_r))
            batch_losses[1].append(float(loss_g))
            batch_losses[2].append(float(loss_b))
            
            loss_r.backward()
            optimizer_r.step()
            optimizer_r.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()
            loss_b.backward()
            optimizer_b.step()
            optimizer_b.zero_grad()
            batch_num+=1
    
    #Validation Loop After Each Epoch
    model_r.eval()
    model_g.eval()
    model_b.eval()
    r_valid = 0
    b_valid = 0
    g_valid = 0
    with torch.no_grad():
        for img_name, lr_r_img, hr_r_img, lr_g_img, hr_g_img, lr_b_img, hr_b_img in v_loaded:
            pred_r = model_r(lr_r_img)
            pred_g = model_g(lr_g_img)
            pred_b = model_b(lr_b_img)
            
            r_valid += torch.nn.functional.mse_loss(pred_r,hr_r_img, reduction="sum")
            g_valid += torch.nn.functional.mse_loss(pred_g,hr_g_img, reduction="sum")
            b_valid += torch.nn.functional.mse_loss(pred_b,hr_b_img, reduction="sum")
    
    r_valid = r_valid / len(v_loaded)
    g_valid = g_valid / len(v_loaded)
    b_valid = b_valid / len(v_loaded)

    print("RED Losses: " + str(batch_losses[0]))
    print("BLUE Losses: " + str(batch_losses[1]))
    print("GREEN Losses: " + str(batch_losses[2]))
    print("EPOCH: " + str(epoch) + " Validation Loss (R,G,B): " + str(float(r_valid)), str(float(g_valid)), str(float(b_valid)))
torch.save(model_r.state_dict(), 'model_r_weights.pth')
torch.save(model_g.state_dict(), 'model_g_weights.pth')
torch.save(model_b.state_dict(), 'model_b_weights.pth')