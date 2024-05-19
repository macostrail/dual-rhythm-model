import pandas as pd
from pathlib import Path
from .logger import setup

import sys
sys.path.append('../')
from config import DATAFRAME, SINUS_NPY_DIR, VPC_NPY_DIR

logger = setup("DEBUG")

def validate_dataset():
    df = pd.read_excel(DATAFRAME, engine='openpyxl')
    orig2_df = df[~df['Orig2'].isna()]
    orig5_df = df[~df['Orig5'].isna()]

    logger.debug(f'Found {orig2_df.shape[0]} samples of orig2 in dataframe')
    logger.debug(f'Found {orig5_df.shape[0]} samples of orig5 in dataframe')
    
    sinus_ids = [path.stem for path in SINUS_NPY_DIR.glob('*.npy')]
    vpc_ids = [path.stem for path in VPC_NPY_DIR.glob('*.npy')]
    
    logger.debug(f'Found {len(sinus_ids)} samples of sinus ecg npy files')
    logger.debug(f'Found {len(vpc_ids)} samples of vpc ecg npy files')

    if set(vpc_ids) != set(sinus_ids):
        logger.debug('Mismatch of npy samples')
