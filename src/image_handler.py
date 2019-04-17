from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt


def load_image_as_tensor(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        image = transform(image).float()
    else:
        image.float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image

def plot_image_tensor(image_tensor):
    plt.figure()
    plt.imshow(image_tensor[0].permute(1, 2, 0))
    plt.show()
