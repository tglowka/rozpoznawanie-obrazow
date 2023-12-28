from torchvision import transforms


class FaceAgeTransforms:
    def __init__(self) -> None:
        pass

    def random_perspective(
        self,
        distortion_scale: int,
    ):
        transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=1)
        return transform

    def random_rotation(self, degrees):
        transform = transforms.RandomRotation(degrees=degrees)
        return transform

    def random_affine(self, scale):
        transform = transforms.RandomAffine(degrees=0, scale=scale)
        return transform

    def elastic_transform(self, alpha, sigma):
        transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma)
        return transform

    def color_jiter(self, brightness=0, contrast=0, saturation=0, hue=0):
        transform = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        return transform

    def gaussian_blur(self, kernel_size, sigma):
        transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        return transform

    def random_adjust_sharpness(self, sharpness_factor):
        transform = transforms.RandomAdjustSharpness(
            sharpness_factor=sharpness_factor, p=1
        )
        return transform

    def random_horizontal_flip(self):
        transform = transforms.RandomHorizontalFlip(p=1)
        return transform

    def create_transforms(self):
        transforms = [
            (
                self.random_rotation(degrees=(-30, -30)),
                "self.random_rotation(degrees=(-30, -30))",
            ),
            (
                self.random_perspective(distortion_scale=0.25),
                "self.random_perspective(distortion_scale=0.25)",
            ),
            (
                self.random_rotation(degrees=(30, 30)),
                "self.random_rotation(degrees=(30, 30))",
            ),
            (
                self.random_affine(scale=(0.75, 0.75)),
                "self.random_affine(scale=(0.75, 0.75))",
            ),
            (
                self.random_affine(scale=(1.25, 1.25)),
                "self.random_affine(scale=(1.25, 1.25))",
            ),
            (
                self.elastic_transform(alpha=100.0, sigma=5.0),
                "self.elastic_transform(alpha=100.0, sigma=5.0)",
            ),
            (
                self.color_jiter(brightness=(1.5, 1.5)),
                "self.color_jiter(brightness=(1.5, 1.5))",
            ),
            (
                self.color_jiter(contrast=(1.5, 1.5)),
                "self.color_jiter(contrast=(1.5, 1.5))",
            ),
            (
                self.color_jiter(saturation=(1.5, 1.5)),
                "self.color_jiter(saturation=(1.5, 1.5))",
            ),
            (self.color_jiter(hue=(0.3, 0.3)), "self.color_jiter(hue=(0.3, 0.3))"),
            (
                self.gaussian_blur(kernel_size=(5, 9), sigma=5),
                "self.gaussian_blur(kernel_size=(5, 9), sigma=5)",
            ),
            (self.random_horizontal_flip(), "self.random_horizontal_flip()"),
        ]
        return transforms
