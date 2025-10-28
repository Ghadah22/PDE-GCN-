import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


# -------------------- helpers --------------------

def _merge_llm_covariates(df_main: pd.DataFrame,
                          cov_path: str,
                          time_col: str = "date",
                          keep_cols=None) -> pd.DataFrame:
    """
    Left-join df_main with an LLM-derived covariate CSV on `time_col`.
    - keep_cols: optional list of covariate column names to keep from cov file
                 (if None, keeps all extra columns except time_col)
    Returns a new dataframe with extra covariate cols appended at the end.
    """
    if (cov_path is None) or (not os.path.exists(cov_path)):
        return df_main

    cov = pd.read_csv(cov_path)
    if time_col not in df_main.columns or time_col not in cov.columns:
        print(f"[llm_cov] WARNING: `{time_col}` missing in one of the files; skip merge.")
        return df_main

    # robust datetime
    df_main = df_main.copy()
    cov = cov.copy()
    df_main[time_col] = pd.to_datetime(df_main[time_col])
    cov[time_col] = pd.to_datetime(cov[time_col])

    # compute extra set
    if keep_cols is None:
        extra_cols = [c for c in cov.columns if c != time_col]
    else:
        extra_cols = [c for c in keep_cols if c in cov.columns]

    if not extra_cols:
        print("[llm_cov] No covariate columns found to merge; skipping.")
        return df_main

    before_cols = set(df_main.columns)
    merged = df_main.merge(cov[[time_col] + extra_cols], on=time_col, how="left")
    added = [c for c in merged.columns if c not in before_cols]
    print(f"[llm_cov] merged {len(added)} extra columns: {added[:8]}{'...' if len(added)>8 else ''}")
    # fill NaNs in added covs with 0 (safe default)
    merged[added] = merged[added].fillna(0)
    return merged


def _build_time_markers(df_time: pd.DataFrame, timeenc: int, freq: str) -> np.ndarray:
    """
    df_time: DataFrame with a 'date' column already sliced [border1:border2]
    Returns: numpy array for time marks
    """
    df_time = df_time.copy()
    df_time['date'] = pd.to_datetime(df_time['date'])

    if timeenc == 0:
        # vectorized accessors (avoids apply and warnings)
        df_time['month'] = df_time['date'].dt.month
        df_time['day'] = df_time['date'].dt.day
        df_time['weekday'] = df_time['date'].dt.weekday
        df_time['hour'] = df_time['date'].dt.hour
        # minute only used in ETT_minute class (handled there)
        data_stamp = df_time.drop(columns=['date']).values
    else:
        data_stamp = time_features(pd.to_datetime(df_time['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp


# -------------------- ETT Hourly --------------------

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None,
                 use_llm_cov: int = 0, llm_cov_path: str = None,
                 llm_keep_cols=None):
        """
        If use_llm_cov=1 and llm_cov_path provided, left-join extra covariates on `date`
        and append them as additional channels to data_x/data_y.
        """
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.use_llm_cov = int(use_llm_cov)
        self.llm_cov_path = llm_cov_path
        self.llm_keep_cols = llm_keep_cols

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Optional: merge LLM covariates on 'date'
        if self.use_llm_cov and self.llm_cov_path:
            df_raw = _merge_llm_covariates(df_raw, self.llm_cov_path, time_col="date",
                                           keep_cols=self.llm_keep_cols)

        # standard ETTh1 splits
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # choose features
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]  # everything except date
            df_data = df_raw[cols_data]
        else:  # 'S'
            if self.target not in df_raw.columns:
                raise ValueError(f"[ETT_hour] target '{self.target}' not in columns.")
            df_data = df_raw[[self.target]]

        # scale on the **training portion**
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        # time markers
        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_markers(df_stamp, self.timeenc, self.freq)

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]
        self.data_stamp = data_stamp

        print(f"[ETT_hour] {self.data_path} features={self.features} "
              f"-> channels: {self.data_x.shape[1]} (set enc_in accordingly)")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# -------------------- ETT Minute --------------------

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 seasonal_patterns=None,
                 use_llm_cov: int = 0, llm_cov_path: str = None,
                 llm_keep_cols=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.use_llm_cov = int(use_llm_cov)
        self.llm_cov_path = llm_cov_path
        self.llm_keep_cols = llm_keep_cols

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.use_llm_cov and self.llm_cov_path:
            df_raw = _merge_llm_covariates(df_raw, self.llm_cov_path, time_col="date",
                                           keep_cols=self.llm_keep_cols)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            if self.target not in df_raw.columns:
                raise ValueError(f"[ETT_minute] target '{self.target}' not in columns.")
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        # time markers
        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            df_stamp['minute'] = (df_stamp['date'].dt.minute // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]
        self.data_stamp = data_stamp

        print(f"[ETT_minute] {self.data_path} features={self.features} "
              f"-> channels: {self.data_x.shape[1]} (set enc_in accordingly)")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# -------------------- Custom CSV --------------------

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None,
                 use_llm_cov: int = 0, llm_cov_path: str = None,
                 llm_keep_cols=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        self.use_llm_cov = int(use_llm_cov)
        self.llm_cov_path = llm_cov_path
        self.llm_keep_cols = llm_keep_cols

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Optional: merge LLM covariates on 'date'
        if self.use_llm_cov and self.llm_cov_path:
            df_raw = _merge_llm_covariates(df_raw, self.llm_cov_path, time_col="date",
                                           keep_cols=self.llm_keep_cols)

        # Robust target handling
        cols_all = list(df_raw.columns)
        if 'date' not in cols_all:
            raise ValueError("[Custom] CSV must contain a 'date' column.")

        if self.target not in cols_all:
            # fall back: use last non-date numeric column
            numeric_cols = [c for c in cols_all if c != 'date' and pd.api.types.is_numeric_dtype(df_raw[c])]
            if not numeric_cols:
                raise ValueError(f"[Custom] target '{self.target}' missing and no numeric columns to fall back.")
            fallback = numeric_cols[-1]
            print(f"[Custom] WARNING: target '{self.target}' not found; using last numeric column '{fallback}' instead.")
            self.target = fallback

        # Reorder columns: date, other_features..., target
        cols = list(df_raw.columns)
        cols.remove('date')
        if self.target in cols:
            cols.remove(self.target)
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 70/10/20 splits (train/val/test by default)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # feature selection
        if self.features in ['M', 'MS']:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        # scale on train slice only
        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data_all = self.scaler.transform(df_data.values)
        else:
            data_all = df_data.values

        # time markers
        df_stamp = df_raw[['date']].iloc[border1:border2].copy()
        data_stamp = _build_time_markers(df_stamp, self.timeenc, self.freq)

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]
        self.data_stamp = data_stamp

        print(f"[Custom] {self.data_path} features={self.features} target={self.target} "
              f"-> channels: {self.data_x.shape[1]} (set enc_in accordingly)")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# -------------------- PEMS (unchanged) --------------------

class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# -------------------- Solar (unchanged) --------------------

class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.seq_len, self.label_len, self.pred_len = size
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data_all = self.scaler.transform(df_data)
        else:
            data_all = df_data

        self.data_x = data_all[border1:border2]
        self.data_y = data_all[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
