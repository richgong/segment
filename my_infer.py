from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_array = np.array(img)
    return img_array

def infer():
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    image = get_image('https://www.islandmovers.com/wp-content/uploads/2020/04/how-to-move-with-a-fridge-full-of-food-300x290.jpg')
    print("Got image:", image)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("Got SAM", mask_generator, mask_generator.predictor.model.device)
    masks = mask_generator.generate(image)
    print("Got masks:", masks)

if __name__ == '__main__':
    infer()
