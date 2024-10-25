import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage

def predict_and_save(generator, input_image, epoch,device,transforms, save_path="predictions"):
    
    generator.eval()
    
    

    input_image = input_image.convert("RGB")
    input_tensor = transforms(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = generator(input_tensor.to(device))  
        
    prediction = (prediction.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  
    output_image = ToPILImage()(prediction.cpu())
    # output_image.save(f"{save_path}/epoch_{epoch + 1}.png")

    generator.train()
    return output_image
