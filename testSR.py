import SRLUT
import PIL.Image
import torch
import torchvision

table_obj = SRLUT.SRLUT(4, 2, 4, 2**4)
c = ["R", "G", "B"]
# for channel in range(len(c)):
#     model = torch.load("model_" + c[channel] + ".pth", map_location=torch.device("cpu"))
#     #model = torch.jit.load("model_" + c[channel] + ".pt")
#     table_obj.produce_tables(model, c[channel])

srlut=[]
for channel in range(len(c)):
    f = open(c[channel] + "_table.json")
    srlut.append(table_obj.decode_table(f))
    f.close()

img = PIL.Image.open("testImage.png")
img_r = torch.tensor(list(img.getdata(0))).view(img.size[1], img.size[0]).tolist()
img_g = torch.tensor(list(img.getdata(1))).view(img.size[1], img.size[0]).tolist()
img_b = torch.tensor(list(img.getdata(2))).view(img.size[1], img.size[0]).tolist()
img.close()

lookup = [img_r, img_g, img_b]

guess_r = torch.tensor((table_obj.test_table(lookup[0], srlut[0])))
guess_g = torch.tensor((table_obj.test_table(lookup[1], srlut[1])))
guess_b = torch.tensor((table_obj.test_table(lookup[2], srlut[2])))
composite_rgb = torch.zeros(3, guess_r.shape[0],guess_r.shape[1])
composite_rgb[0] = guess_r
composite_rgb[1] = guess_g
composite_rgb[2] = guess_b
guess_img = torchvision.transforms.functional.to_pil_image(composite_rgb)
f = open("finally.png", "wb")
guess_img.save(f)
f.close()
