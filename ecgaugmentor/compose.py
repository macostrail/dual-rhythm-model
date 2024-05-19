class Compose:
    """
    Compose several transforms together.

    ex)
    '''
    transform = aug.Compose([
        aug.BaseLiner(),
        aug.LowHighPass(lowpass_range=(80, 110), highpass_range=(0.03, 0.05)),
        aug.StrechTime((1.1, 1.5)),
        aug.StrechVoltage((1.1, 1.5)),
        aug.RandomNoise(low_fq_hz_max=0.1, low_fq_mv_max=1,
                        high_fq_hz_max=150, high_fq_mv_max=0.03),
        aug.TimeShift((-30, 30))
    ])
    transform(ecg)
    ```

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
