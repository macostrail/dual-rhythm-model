import cv2
import numpy as np
import psd_tools
from psd_tools.api.layers import PixelLayer
from pathlib import Path


class EcgDigitizer:
    """
    Outputs standardized 12-lead ECG waveform from PSD format.
    The layers of the PSD are as follows:
    I_s, I_v, II_s, II_v, III_s, III_v, aVR_s, aVR_v, aVL_s, aVL_v, aVF_s, aVF_v,
    V1_s, V1_v, V2_s, V2_v, V3_s, V3_v, V4_s, V4_v, V5_s, V5_v, V6_s, V6_v
    s: sinus rhythm wave
    v: VPC wave
    ref_l: reference for limb leads
        Vertical: 1mV, Horizontal: 200msec
    ref_t: reference for precordial leads (if exists)
        Vertical: 1mV, Horizontal: 200msec

    Conversion process steps:
    1. Obtain waveform images of each layer from the PSD
    2. Convert from waveform images to one-dimensional (complement missing parts, zero the baseline)
    3. Align the position and width of peaks within limb leads and precordial leads using offset information for each layer
    4. Align the position and width of peaks between precordial leads and limb leads
    5. Extend the baseline to match the output sampling rate and duration
    """
    def __init__(self, out_samlping_rate=500, out_sec=0.8, ref_msec=200):
        """
        out_samlping_rate
        out_sec: output wave seconds
        ref_msec: time width of the reference(msec)
        """
        self.LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                      'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.out_samlping_rate = out_samlping_rate
        self.ref_msec = ref_msec
        self.out_sec = out_sec

    def from_psd(self, psd: PixelLayer):
        """
        Get ECG waveform from PSD
        Returns:
            {
                's': sinus rythm arr(np.ndarray): (12, length),
                'v': vpc arr(np.ndarray): (12, length),
            }
        """
        assert self.check_psd_file, 'psd file is not valid.'
        self.set_layer_name_dict(psd)
        self.set_ref_config(psd)
        waves = {}
        for wave_type in ['s', 'v']:
            waves_d = self.get_waves_dict(psd, wave_type)
            waves_d = self.inner_extend(waves_d)
            waves_d = self.adjust_limb_precord(waves_d)
            waves[wave_type] = self.outer_extend(waves_d)
        return waves

    def set_ref_config(self, psd):
        ref_l_mv, ref_sec = \
            psd[self.layer_names_dict['ref_l']].numpy().shape[:2]
        sec_magnifi = self.out_samlping_rate / (ref_sec * (1000/self.ref_msec))
        self.ref_l_mv = ref_l_mv
        self.sec_magnifi = sec_magnifi

    def set_layer_name_dict(self, psd):
        """Get the layer name dictionary from the PSD file."""
        layer_names_dict = {}
        for i, layer in enumerate(psd):
            layer_names_dict[layer.name] = i
        self.layer_names_dict = layer_names_dict

    def check_psd_file(self, psd_file: str):
        """
        Get ECG waveform from PSD and check if
        the correct layer name is attached to the PSD file.
        """
        psd = psd_tools.PSDImage.open(psd_file)
        layer_names_dict = {}
        for i, layer in enumerate(psd):
            layer_names_dict[layer.name] = i
        ret = []
        for lead in self.LEADS:
            for t in ['s', 'v']:
                if f'{lead}_{t}' not in layer_names_dict.keys():
                    ret.append(f'{lead}_{t}')
        if 'ref_l' not in layer_names_dict.keys():
            ret.append('ref_l')
        if ret:
            return ','.join(ret) + 'is not included.'
        return True

    def get_waves_dict(self, psd, wave_type):
        waves_d = []
        error_logs = []
        for lead in self.LEADS:
            layer_idx = self.layer_names_dict[f'{lead}_{wave_type}']
            img = psd[layer_idx].numpy()
            img = self.clear_erasor_trace(img)[..., 3]
            if point:= self.has_noise(img):
                error_logs.append({
                    'lead': lead, 'wave_type': wave_type, 'point': point
                })
                continue
            waves_d.append(
                {'wave': self.img2wave(img),
                 'offset': psd[layer_idx].offset}
            )
        if error_logs:
            raise Exception(', '.join(
                [f'noise at ({d["point"][0]}, {-d["point"][1]})' +
                 f'from main component in {d["lead"]}_{d["wave_type"]}'
                 for d in error_logs]))
        return waves_d

    def col2val(self, line: np.ndarray):
        """
        line: 1D array. Get median from the unit time column.
        """
        line = np.flip(line)
        median = np.median(np.where(line>0)[0])
        if np.isnan(median):
            return np.nan
        return int(median)

    def img2wave(self, img):
        """ Get waveform from image """
        tgt_width = int(img.shape[1] * self.sec_magnifi)
        img = cv2.resize(img, dsize=(tgt_width, img.shape[0]))
        wave = []
        for i in range(img.shape[1]):
            wave.append(self.col2val(img[:, i]))
        wave = np.array(wave)
        wave = self.interpole(wave) / self.ref_l_mv
        return self.baseliner(wave)

    def nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def interpole(self, wave):
        nans, x= self.nan_helper(wave)
        wave[nans]= np.interp(x(nans), x(~nans), wave[~nans])
        return wave

    def calc_inner_edge_offset(self, waves_d):
        ret = {}
        # for limb leads
        left_offset = np.min([d['offset'][0] for d in waves_d[:6]])
        right_offset = np.max([d['offset'][0]+d['wave'].shape[0] for d in waves_d[:6]])
        ret['limb'] = (left_offset, right_offset)

        # for precordial leads
        left_offset = np.min([d['offset'][0] for d in waves_d[6:]])
        right_offset = np.max([d['offset'][0]+d['wave'].shape[0] for d in waves_d[6:]])
        ret['precord'] = (left_offset, right_offset)
        return ret

    def baseliner(self, wave):
        start_val = wave[0]
        return wave - start_val

    def inner_extend(self, waves_d):
        """
        Matching the position of the peaks in the limb leads and precordial leads.
        """
        edge_d = self.calc_inner_edge_offset(waves_d)
        left_offset, right_offset = edge_d['limb']
        for d in waves_d[:6]:
            inner_extend_px = (
                d['offset'][0] - left_offset,
                right_offset - (d['offset'][0]+d['wave'].shape[0])
            )
            d['wave'] = self.extend_edge(d['wave'], *inner_extend_px)

        left_offset, right_offset = edge_d['precord']
        for d in waves_d[6:]:
            inner_extend_px = (
                d['offset'][0] - left_offset,
                right_offset - (d['offset'][0]+d['wave'].shape[0])
            )
            d['wave'] = self.extend_edge(d['wave'], *inner_extend_px)
        assert all([d['wave'].shape[0] == waves_d[:6][0]['wave'].shape[0] for d in waves_d[:6]])
        assert all([d['wave'].shape[0] == waves_d[6:][0]['wave'].shape[0] for d in waves_d[6:]])
        return waves_d

    def outer_extend(self, waves_d):
        """
        Align the position and width of peaks between precordial leads and limb leads.
        Returns:
            np.array: 12-lead waveform (shape: (12, width))
        """
        width = int(self.out_sec * self.out_samlping_rate)
        waves = np.array([d['wave'] for d in waves_d])
        extend_px = width - waves.shape[1]
        peak = int(np.mean(np.argmax(np.abs(waves), axis=1)))
        left_shift, right_shift = 0, 0
        if peak < width / 2:
            left_shift += min(width//2 - peak, extend_px)
            if left_shift < extend_px:
                margin = extend_px - left_shift
                left_shift += margin//2
                right_shift += margin - margin//2
        else:
            right_shift += min(peak - width//2, extend_px)
            if right_shift < extend_px:
                margin = extend_px - right_shift
                left_shift += margin//2
                right_shift += margin - margin//2
        return np.apply_along_axis(
            lambda x: self.extend_edge(x, left_shift, right_shift),
            axis=1,
            arr=waves
        )

    def adjust_limb_precord(self, waves_d):
        """
        Adjustment of the position and width of peaks between limb leads and precordial leads.
        """
        limb_waves = np.array([d['wave'] for d in waves_d[:6]])
        precord_waves = np.array([d['wave'] for d in waves_d[6:]])
        peak_l = np.where(np.abs(limb_waves)==np.max(np.abs(limb_waves)))[1][0]
        peak_t = np.where(np.abs(precord_waves)==np.max(np.abs(precord_waves)))[1][0]
        if peak_t > peak_l:
            shift_px = int(peak_t - peak_l)
            for d in waves_d[6:]:
                d['wave'] = d['wave'][shift_px:]
        else:
            shift_px = int(peak_l - peak_t)
            for d in waves_d[:6]:
                d['wave'] = d['wave'][shift_px:]

        lt_dif = waves_d[0]['wave'].shape[0] - waves_d[6]['wave'].shape[0]
        if lt_dif > 0:
            for d in waves_d[6:]:
                d['wave'] = self.extend_edge(d['wave'], right_px=lt_dif)
        else:
            for d in waves_d[:6]:
                d['wave'] = self.extend_edge(d['wave'], right_px=-lt_dif)
        assert all([d['wave'].shape[0] == waves_d[0]['wave'].shape[0] for d in waves_d])
        return waves_d

    def extend_edge(self, wave, left_px: int=0, right_px: int=0):
        start_val = wave[0]
        end_val = wave[-1]
        wave = np.concatenate((
            [start_val for _ in range(left_px)],
            wave,
            [end_val for _ in range(right_px)]
        ))
        return wave

    def has_noise(self, img):
        """Checking for noise in the image."""
        _, bin_img = cv2.threshold(np.uint8(img*255), 1e-7, 255, cv2.THRESH_BINARY)
        obj_n, _, stats, weights = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
        if obj_n == 2:
            # backgorund　included
            return False
        else:
            sort_ind = np.argsort(stats[:, 4])  # ascend by area
            weights = weights[sort_ind]
            sorted_stats = stats[sort_ind]
            if sorted_stats[0, 4] > 100:  # separated trace
                return False
            noise_weight = weights[0]
            main_weight = weights[-2]
            return (round(noise_weight[0]-main_weight[0]),
                    round(noise_weight[1]-main_weight[1]))

    def clear_erasor_trace(self, img: np.ndarray):
        """Removing the eraser trace.
        Returns:
            img (np.ndarray): shape:(height, width, 4)
        """
        loc = np.where((0 < img[..., 3]) & (img[..., 3] < 0.2))
        img[loc] = np.array([1, 1, 1, 0])
        return img


def main(args):
    ecgdigitizer = EcgDigitizer()
    error_files = []
    for psd_file in Path(args.psd_dir).glob('*.psd'):
        try:
            pid = Path(psd_file).stem
            output_dir = Path(args.output_dir)
            if (output_dir / f'{pid}_s.npy').exists():
                continue

            psd = psd_tools.PSDImage.open(psd_file)
            waves = ecgdigitizer.from_psd(psd)
            np.save(output_dir / f'{pid}_s', waves['s'])
            np.save(output_dir / f'{pid}_v', waves['v'])
        except Exception as e:
            error_files.append((psd_file.name, e))

    for f, e in error_files:
        print(f, e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--psd_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
