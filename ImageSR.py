import torch
import sys
import os
import PIL.Image
import ImageDataset
import ImageSRModel
import torch.utils.data
import SRLUT

train_dir = sys.argv[1]
target_dir = sys.argv[2]
valid_lr_dir = sys.argv[3]
valid_hr_dir = sys.argv[4]
resume = sys.argv[5]
train_dataset = os.listdir(train_dir)
target_dataset = os.listdir(target_dir)
valid_lr_dataset = os.listdir(valid_lr_dir)
valid_hr_dataset = os.listdir(valid_hr_dir)

check_file_name = "RecentEpochSaveState.pt"

lr_h = 510 
lr_w = 279
load = False

if (torch.cuda.is_available()):
    dev = "cuda"
else:
    dev = "cpu"

if (resume == "y" or resume == "Y"):
    load = True
    check_file = torch.load(check_file_name)

FACTOR = 4
RF = 4
CONV_KERNEL = 2 #Kernel is 2x2
TOTAL_ITERATIONS = 800
BATCH_SIZE = 1
BATCHES = int(len(train_dataset) / BATCH_SIZE)
EPOCHS = int(TOTAL_ITERATIONS / len(train_dataset))

if (load):
    start_epoch = check_file["epoch"]
    start_channel = check_file["start_channel"]
else:
    start_epoch = 0
    start_channel = 0

c = ["R", "G", "B"]
for channel in range(start_channel, 3):
    d_set = ImageDataset.ImageDataset(train_dataset, train_dir, target_dir, lr_h, lr_w, channel, FACTOR, dev)
    d_loaded = torch.utils.data.DataLoader(d_set, batch_size = BATCH_SIZE, shuffle = True)

    v_set = ImageDataset.ImageDataset(valid_lr_dataset, valid_lr_dir, valid_hr_dir, lr_h, lr_w, channel, FACTOR, dev)
    v_loaded = torch.utils.data.DataLoader(v_set, batch_size = 2*BATCH_SIZE)

    model = ImageSRModel.ImageSRModel(FACTOR).to(dev)
    optimizer = torch.optim.Adam(model.parameters())

    if (load and channel == start_channel):
        model.load_state_dict(check_file["model_state"])
        optimizer.load_state_dict(check_file["optim_state"])
    
    for epoch in range(start_epoch, EPOCHS):
        batch_num = 0
        batch_losses = []
        model.train()
        
        for lr_img, hr_img in d_loaded:
            print("EPOCH: " + str(epoch) + " BATCH: " + str(batch_num))
            pred = model(lr_img)
                        
            loss = torch.nn.functional.mse_loss(pred,hr_img, reduction="sum")
            
            batch_losses.append(float(loss))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_num+=1
        del lr_img, hr_img
        
        #Validation Loop After Each Epoch
        model.eval()
        
        valid = 0
        
        with torch.no_grad():
            for lr_img, hr_img in v_loaded:
                pred = model(lr_img)
                                
                valid += torch.nn.functional.mse_loss(pred,hr_img, reduction="sum")
                
            del lr_img, hr_img
        valid = valid / len(v_loaded)
        
        print("EPOCH: " + str(epoch) + " Validation Loss (R,G,B): " + str(float(valid)))
        print(c[channel] + " Batch Losses: " + str(batch_losses))
        f = open(check_file_name, "w")
        torch.save({"start_epoch":epoch+1, "model_state":model.state_dict(), "optim_state": optimizer.state_dict(), "start_channel":channel}, f)
        f.close()
    model_export = torch.jit.script(model)
    m_file = str("model_" + c[channel] + ".pt")
    model_export.save(m_file)