
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import argparse
import sys
from torchvision import transforms
from tqdm import tqdm
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from progress.bar import IncrementalBar
from PIL import Image
from model.generator import UnetGenerator
from model.discriminator import ConditionalDiscriminator
from model.losses import GeneratorLoss,DiscriminatorLoss
from utils.logger import Logger
from utils.initialize_weights import initialize_weights
from data.datasets import Maps
from predict import predict_and_save

parser = argparse.ArgumentParser(prog='top', description='Train Pix2Pix')
parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--dataset", type=str, default="maps", help="Name of the dataset: ['facades', 'maps', 'cityscapes']")
parser.add_argument("--batch_size", type=int, default=64, help="Size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adams learning rate")

# Avoids unrecognized arguments in Jupyter
args = parser.parse_args(args=[])


device=('cuda:0' if torch.cuda.is_available else 'cpu')

transforms =transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x:(x-0.5)/0.5)

])


#Define the model

generator=UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)
generator.apply(initialize_weights)
discriminator.apply(initialize_weights)

g_optimizer=torch.optim.Adam(generator.parameters(),lr=args.lr,betas=(0.5,0.999))
d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=args.lr,betas=(0.5,0.999))

g_loss=GeneratorLoss(alpha=100)
d_loss=DiscriminatorLoss()

dataset=Maps(root='.',transform=transforms,download=True,mode='train')

dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
logger=Logger(filename=args.dataset)


for epoch in tqdm(range(args.epochs)):
  generator.train()
  discriminator.train()
  ge_loss=0
  de_loss=0

  start=time.time()
  bar = IncrementalBar(f'[Epoch {epoch+1}/{args.epochs}]', max=len(dataloader))

  for x,real in dataloader:
    x=x.to(device)
    real=real.to(device)

    fake=generator(x).detach()
    fake_pred=discriminator(fake,x)
    real_pred=discriminator(real,x)
    d_loss_val=d_loss(fake_pred,real_pred)

    d_optimizer.zero_grad()
    d_loss_val.backward()
    d_optimizer.step()

    fake = generator(x)
    fake_pred=discriminator(fake,x)
    g_loss_val=g_loss(fake,real,fake_pred)

    g_optimizer.zero_grad()
    g_loss_val.backward()
    g_optimizer.step()

    ge_loss+=g_loss_val.item()
    de_loss+=d_loss_val.item()


  bar.finish()
  g_loss_val = ge_loss/len(dataloader)
  d_loss_val = de_loss/len(dataloader)
  # count timeframe
  end = time.time()
  tm = (end - start)
  # logger.add_scalar('generator_loss', g_loss, epoch+1)
  # logger.add_scalar('discriminator_loss', d_loss, epoch+1)
  # logger.save_weights(generator.state_dict(), 'generator')
  # logger.save_weights(discriminator.state_dict(), 'discriminator')
  print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch+1, args.epochs, g_loss_val, d_loss_val, tm))

  if(epoch%50==0):
    torch.save(generator,f'generator{epoch}.pth')

logger.close()
print('End of training process!')

torch.save(generator.state_dict(), f'generator.pth')
torch.save(discriminator.state_dict(), f'discriminator.pth')

# torch.save(g_optimizer.state_dict(), f'g_optimizer_epoch_{epoch}.pth')
# torch.save(d_optimizer.state_dict(), f'd_optimizer_epoch_{epoch}.pth')

config = {
    "epochs": args.epochs,
    "learning_rate": args.lr,
    "batch_size": args.batch_size,
    "alpha": g_loss.alpha,
}
with open('config.json', 'w') as f:
    json.dump(config, f)


folder_path='maps/val'
path_list=[]
input_img=[]
target_img=[]
output_img=[]
for filename in os.listdir(folder_path):
  path_list.append(os.path.join(folder_path,filename))

generator = UnetGenerator().to(device)  # Ensure UnetGenerator is defined/imported
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

for ind,i in enumerate(path_list):

  img=Image.open(i)
  width,height=img.size

  input_img.append(img.crop((0,0,width//2,height)))
  target_img.append(img.crop((width//2,0,width,height)))
  output_img.append(predict_and_save(generator, img.crop((0,0,width//2,height)) , epoch,device,transforms))



score = 0.0
for i in range(len(output_img)):
    # Convert both target and output PIL images to NumPy arrays
    target_img_np = np.array(target_img[i])
    output_img_np = np.array(output_img[i])
    
    # Ensure both images have the same shape and are color (3 channels)
    if target_img_np.shape != output_img_np.shape:
        output_img_np = cv2.resize(output_img_np, (target_img_np.shape[1], target_img_np.shape[0]))

    # Compute SSIM score
    ssim_score, _ = ssim(target_img_np, output_img_np, full=True, channel_axis=-1)
    score += ssim_score

# Average SSIM score across all image pairs
average_ssim_score = score / len(output_img)
print("Average SSIM:", average_ssim_score)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))  # 3 rows, 3 columns

# Manually assign images to each subplot
axes[0, 0].imshow(input_img[362])
axes[0, 0].axis('off')  # Hide axes

axes[0, 1].imshow(input_img[679])
axes[0, 1].axis('off')

axes[0, 2].imshow((input_img[221]))
axes[0, 2].axis('off')

axes[1, 0].imshow(target_img[362])
axes[1, 0].axis('off')

axes[1, 1].imshow(target_img[679])
axes[1, 1].axis('off')

axes[1, 2].imshow(target_img[221])
axes[1, 2].axis('off')

axes[2, 0].imshow(output_img[362])
axes[2, 0].axis('off')

axes[2, 1].imshow(output_img[679])
axes[2, 1].axis('off')

axes[2, 2].imshow(output_img[221])
axes[2, 2].axis('off')

# Adjust spacing
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Show the plot
plt.show()