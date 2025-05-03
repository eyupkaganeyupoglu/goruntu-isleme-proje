    def convert_rgb_to_hsv(self, image):
        hsv_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j] / 255.0
                cmax = max(r, g, b)
                cmin = min(r, g, b)
                delta = cmax - cmin

                if delta == 0:
                    h = 0
                elif cmax == r:
                    h = (60 * ((g - b) / delta) + 360) % 360
                elif cmax == g:
                    h = (60 * ((b - r) / delta) + 120) % 360
                else:
                    h = (60 * ((r - g) / delta) + 240) % 360

                s = 0 if cmax == 0 else delta / cmax
                v = cmax

                hsv_image[i, j] = [int(h / 2), int(s * 255), int(v * 255)]
        return hsv_image
    
    def convert_rgb_to_ycbcr(self, image):
        ycbcr_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r, g, b = image[i, j]
                y  =  0.299 * r + 0.587 * g + 0.114 * b
                cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
                cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 128
                ycbcr_image[i, j] = [int(y), int(cb), int(cr)]
        return ycbcr_image