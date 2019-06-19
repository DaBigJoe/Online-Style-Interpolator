from src.interpolator.interpolate_styles import NStyleInterpolator
import torch
from src.interpolator.image_handler import load_image_as_tensor, transform_256, save_tensor_as_image
from PIL import Image
import base64
import io

TOTAL_NUM_STYLES = 5
NUM_STYLES_USED = 5
NETWORK_PATH = '../../data/networks/model_parameters/0006'


class InterpolationHandler:

    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.interpolator = NStyleInterpolator(TOTAL_NUM_STYLES, NETWORK_PATH, device)
        self.style_idxs = range(NUM_STYLES_USED)

        self.test_image_tensor = load_image_as_tensor('../../data/images/content/venice.jpeg', transform=transform_256)
        self.test_image_tensor = self.test_image_tensor.unsqueeze(0).to(device)

    def interpolate(self, weights):
        interpolated_style_parameters = self.interpolator.interpolate(self.style_idxs, weights)
        t = self.interpolator.render_interpolated_image(interpolated_style_parameters, self.test_image_tensor)
        t = t.squeeze(0).detach()
        t = t.clone().clamp(0, 255).numpy()
        t = t.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(t, 'RGB')
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes)
