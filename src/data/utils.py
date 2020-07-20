import cv2
import numpy as np
import torch


class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def _normalize(self, image):
        image = torch.from_numpy(image).permute(2, 0, 1) / 255.

        mean = torch.as_tensor(self.mean, dtype=torch.float32)
        std = torch.as_tensor(self.std, dtype=torch.float32)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image

    def __call__(self, image):
        return self._normalize(image)


class Letterbox:

    def __init__(self, height: int, width: int, fill_color: int = 0):
        self.height = height
        self.width = width
        self.fill_color = fill_color

    def __call__(self, image):
        image, info = self._letterbox_image(image)
        return image, info

    def _letterbox_image(self, image):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3

        info = {'padding': ((0, 0), (0, 0))}

        dst_h, dst_w = self.height, self.width
        src_h, src_w = image.shape[:2]

        if src_w == dst_w and src_h == dst_h:
            return image, info

        if src_w / dst_w >= src_h / dst_h:
            scale = dst_w / src_w
        else:
            scale = dst_h / src_h

        info['orig_size'] = (src_h, src_w)

        if src_w / dst_w != 1.0 or src_h / dst_h != 1.0:
            image_resized = cv2.resize(image, (int(scale * src_w), int(scale * src_h)), interpolation=cv2.INTER_LANCZOS4, )
            resized_h, resized_w = image_resized.shape[:2]
        else:
            return image, info

        if dst_w == resized_w and dst_h == resized_h:
            return image_resized, info
        else:
            pad_w = (dst_w - resized_w) / 2
            pad_h = (dst_h - resized_h) / 2
            pad = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
            pad_width = ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0))
            info['padding'] = pad_width[:-1]
            image = np.pad(image_resized, pad_width, mode='constant', constant_values=self.fill_color)
            return image, info


class VanillaResize:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        image, info = self._resize(image)
        return image, info

    def _resize(self, image):
        info = {'orig_size': image.shape}
        image = cv2.resize(image, (self.height, self.width))
        return image, info


def perform_fgsm_attack(model, batch, criterion, epsilon, device):
    batch = {key: val.to(device) for key, val in batch.items()}
    batch['image'].requires_grad = True

    pred = model(batch['image'])
    model.zero_grad()
    loss = criterion(pred, batch['label'])
    loss.backward()
    
    batch['image'] = batch['image'] + epsilon * batch['image'].grad.sign()
    batch['image'] = torch.clamp(batch['image'], 0, 1)
    return batch
