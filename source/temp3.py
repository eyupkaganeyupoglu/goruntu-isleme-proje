import numpy as np

def filter_unsharp(self, image, amount=1.0):
    if len(image.shape) == 3:
        image_gray = self.convert_to_gray(image)
    else:
        image_gray = image

    image_gray = image_gray.astype(np.float32)

    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    padded = np.pad(image_gray, ((1, 1), (1, 1)), mode='reflect')
    blurred = np.zeros_like(image_gray)

    height, width = image_gray.shape

    for y in range(1, height + 1):
        for x in range(1, width + 1):
            region = padded[y-1:y+2, x-1:x+2]
            blurred[y-1, x-1] = np.sum(region * kernel)

    mask = image_gray - blurred

    sharpened = image_gray + amount * mask

    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened