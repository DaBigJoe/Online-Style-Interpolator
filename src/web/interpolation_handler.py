from src.interpolator.interpolate_styles import NStyleInterpolator
import torch
from src.interpolator.image_handler import load_image_as_tensor, transform_256
from PIL import Image
import base64
import io

TOTAL_NUM_STYLES = 5
NUM_STYLES_USED = 5
NETWORK_PATH = '../../data/networks/model_parameters/0006'


class InterpolationHandler:

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.interpolator = NStyleInterpolator(TOTAL_NUM_STYLES, NETWORK_PATH, self.device)
        self.style_idxs = range(NUM_STYLES_USED)
        self.image_tensor = None
        self.locked = False

    def update_image(self, path):
        self.image_tensor = load_image_as_tensor(path, transform=transform_256).unsqueeze(0).to(self.device)

    def attempt_interpolation(self, weights):
        image_binary = None
        if not self.locked:
            self.locked = True
            image_binary = self.interpolate(weights)
            self.locked = False
        return image_binary

    def interpolate(self, weights):
        if self.image_tensor is not None:
            interpolated_style_parameters = self.interpolator.interpolate(self.style_idxs, weights)
            t = self.interpolator.render_interpolated_image(interpolated_style_parameters, self.image_tensor)
            t = t.squeeze(0).detach()
            t = t.clone().clamp(0, 255).numpy()
            t = t.transpose(1, 2, 0).astype("uint8")
            img = Image.fromarray(t, 'RGB')
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes)
        return None
