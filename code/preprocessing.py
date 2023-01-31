import cv2
import torch
### image preprocessing function
def prepare_image(path,image_size):
    # import image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    # resize and color
    image = cv2.resize(image, (int(image_size), int(image_size)))
    image = image.astype("float32") / 255.0
    # convert to tensor    
    image = torch.tensor(image)
    image = image.permute(2, 1, 0)
    return image

