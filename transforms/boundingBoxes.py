
# import path library
from pathlib import Path
# import torch library for tensor operations
import torch
# import matplotlib for visualization
import matplotlib.pyplot as plt
# import transformsvs for image transformations
from torchvision.transforms import v2
# import read_image function from torchvision.io
from torchvision.io import read_image
from torchvision import tv_tensors  # we'll describe this a bit later, bare with us
from helpers import plot

# by providing a seed, we ensure that the random number generated is the same each time we run the code
torch.manual_seed(1)

# read the image from the assets folder
img = read_image(str(Path('imgs/') / 'astronaut1.jpg'))

# create 3 bounding boxes to be added to the image
boxes = tv_tensors.BoundingBoxes(
    [
        [15, 10, 370, 510],
        [275, 340, 510, 510],
        [130, 345, 210, 425]
    ],
    format="XYXY", canvas_size=img.shape[-2:])

# sets the bounding box for saving the image to be tight
plt.rcParams["savefig.bbox"] = 'tight'




# create a transform that crops the image to 300x300, flips it horizontally and makes it grayscale
pfTransform = v2.Compose([
    v2.CenterCrop(300),
    v2.RandomHorizontalFlip(p=1),
    v2.Grayscale(num_output_channels=1)
])

# apply the transform to the image any save the transformed image and boxes 
img_transformed, boxes_transformed = pfTransform(img, boxes)
# print the transformed image type, data type and shape
print(f"{type(img_transformed) = }, {img_transformed.dtype = }, {img_transformed.shape = }")

# plot the original and transformed images side by side
# plot(img, img_transformed, img_labels=['Original', 'Transformed'], figsize=(8, 4))

