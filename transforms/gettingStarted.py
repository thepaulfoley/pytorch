
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

# sets the bounding box for saving the image to be tight
plt.rcParams["savefig.bbox"] = 'tight'
# by providing a seed, we ensure that the random number generated is the same each time we run the code
torch.manual_seed(1)



from helpers import plot
# read the image from the assets folder
img = read_image(str(Path('imgs/') / 'astronaut1.jpg'))
# print  the image type, data type and shape
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")

# create a transform that crops the image to 300x300, flips it horizontally and makes it grayscale
pfTransform = v2.Compose([
    v2.CenterCrop(300),
    v2.RandomHorizontalFlip(p=1),
    v2.Grayscale(num_output_channels=1)
])

# apply the transform to the image
img_transformed = pfTransform(img)
# print the transformed image type, data type and shape
print(f"{type(img_transformed) = }, {img_transformed.dtype = }, {img_transformed.shape = }")

# plot the original and transformed images side by side
# plot(img, img_transformed, img_labels=['Original', 'Transformed'], figsize=(8, 4))

