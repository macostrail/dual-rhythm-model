from pathlib import Path

# docker absolute path
DATASET_DIR = Path('/workspace/projects/vpc_v2/dataset')
NPY_DIR = DATASET_DIR / 'input_data'
SINUS_NPY_DIR = NPY_DIR / 'sinus'
VPC_NPY_DIR = NPY_DIR / 'vpc'
DATAFRAME = DATASET_DIR / 'origin_data_summary.xlsx'
PRETRAIN_VPC_NPY = DATASET_DIR / 'pretrain/vpcs.npy'
CKPT_DIR = Path('/workspace/projects/vpc_v2/ckpt')
PROJECT = 'vpc-ai-v2'
DATAFRAME_ID_COLNAME = 'Num'

INPUT_SHAPE = (12, 400)

HARD_LOWPASS_RANGE = (10, 100)
SOFT_LOWPASS_RANGE = (90, 100)

HARD_HIGHPASS_RANGE = (0.01, 0.05)
SOFT_HIGHPASS_RANGE = (0.01, 0.05)

HARD_STRECHTIME_RANGE = (1.01, 2.0)
SOFT_STRECHTIME_RANGE = (1.01, 1.5)

HARD_STRECHVOLTAGE_RANGE = (0.6, 2.5)
SOFT_STRECHVOLTAGE_RANGE = (0.8, 2.0)

HARD_LOW_FQ_HZ_MAX = 0.3
SOFT_LOW_FQ_HZ_MAX = 0.05

HARD_LOW_FQ_MV_MAX = 1.0
SOFT_LOW_FQ_MV_MAX = 0.5

HARD_HIGH_FQ_HZ_MAX = 200
SOFT_HIGH_FQ_HZ_MAX = 100

HARD_HIGH_FQ_MV_MAX = 0.4
SOFT_HIGH_FQ_MV_MAX = 0.2

HARD_TIMESHIFT_RANGE = (-100, 100)
SOFT_TIMESHIFT_RANGE = (-30, 30)
