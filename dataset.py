import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from config import SINUS_NPY_DIR, VPC_NPY_DIR, NPY_DIR, DATAFRAME_ID_COLNAME
from utils.logger import setup
from utils import utils
from typing import Callable

logger = setup("DEBUG")


class EcgDataset(Dataset):
    """
    custom dataset of ECG data
    Args:
        pm:
            pm.in_dims(int): 3 => (1, leads, length)
                            2 => (leads, length)
            pm.target_col(string): col name of outcome
            pm.input_shape(Tuple)
            pm.mode(str): ['s', v', 'sv']
                s: only sinus wave
                v: only vpc wave
                sv: sinus wave + vpc wave
    """
    def __init__(self, df: pd.DataFrame,
                 pm, transform=None, verbose=0):
        self.pm = pm
        self.transform = transform
        self.verbose = verbose
        self.error_file_log = []
        self.input_shape = tuple(map(int, pm.input_shape.split()))

        self._validate_dataset()

        sinus_ids = [path.stem.split('_')[0] for path in SINUS_NPY_DIR.glob('*.npy')]

        self.ids = [str(id_).zfill(6) for id_ in
                    df[DATAFRAME_ID_COLNAME].astype(int).tolist()
                    if str(id_).zfill(6) in sinus_ids]

        X_s, X_v= [], []
        for id in self.ids:
            if 's' in self.pm.mode:
                sinus_ecg = np.load(SINUS_NPY_DIR / f'{id}_s.npy')
                sinus_ecg = utils.to_12leads(sinus_ecg, with_cabrera=pm.cabrera)

                if sinus_ecg.shape != self.input_shape:
                    self.error_file_log.append((id, 'shape mismatch'))
                    continue
                X_s.append(sinus_ecg)
            if 'v' in self.pm.mode:
                vpc_ecg = np.load(VPC_NPY_DIR /  f'{id}_v.npy')
                vpc_ecg = utils.to_12leads(vpc_ecg, with_cabrera=pm.cabrera)
                if vpc_ecg.shape != self.input_shape:
                    self.error_file_log.append((id, 'shape mismatch'))
                    continue
                X_v.append(vpc_ecg)

        if self.verbose:
            print('Numpy array shape is not valid with following files')
            print('\n'.join([t[0] for t in self.error_file_log]))

        self.X_s = np.array(X_s)
        self.X_v = np.array(X_v)
        self.Y = np.array(df[self.pm.target_col].tolist())
        print('X_s.shape', self.X_s.shape)
        print('X_v.shape', self.X_v.shape)

    def _validate_dataset(self):
        sinus_ids = [
            path.stem.split('_')[0] for path in
            SINUS_NPY_DIR.glob('*.npy')
        ]
        vpc_ids = [
            path.stem.split('_')[0] for path in
            VPC_NPY_DIR.glob('*.npy')
        ]
        assert len(sinus_ids) == len(vpc_ids) == len(set(
            [path.stem.split('_')[0] for path in
             NPY_DIR.glob('**/*.npy')]
        ))

        assert self.pm.mode in ['s', 'v', 'sv'], 'please set the mode as s, v, or sv'

    def __getitem__(self, index):
        def f(arr):
            if self.transform:
                arr = self.transform(arr)
            if self.pm.in_dims == 3:
                arr = np.expand_dims(arr, 0)
            return arr

        if self.pm.mode == 's':
            return f(self.X_s[index]), self.Y[index]
        if self.pm.mode == 'v':
            return f(self.X_v[index]), self.Y[index]
        if self.pm.mode == 'sv':
            return f(self.X_s[index]), f(self.X_v[index]), self.Y[index]

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    pass
