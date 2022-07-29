import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
# from torch.nn.utils.rnn import pad_sequence


def load_data(chunk_paths, features, df_target=None, presort=False, ascending=False, shift_value=0):
    """
    chunk_paths: подготовленный список путей к частям датасета
    features: список признаков, которые необходимо оставить в данных
    df_target: pd.DataFrame со значениями таргетов для каждой последовательности
        столбцы: <id> <flag>
    presort: осуществить ли сортировку последовательностей по длине (по возрастанию)
    ascending: способ сортировки элементов последовательностей; по умолчанию - по убыванию,
        т.е. значения с меньшими индексами соответствуют более поздним значениями по времени
    shift_value: значение, на которое нужно увеличить все значения категориальных переменных
    """
    res = []
    for chunk_path in chunk_paths:
        chunk = pd.read_parquet(chunk_path)
        chunk = (
            chunk
            .astype({f: np.int8 for f in features})
            .sort_values(["id", "rn"], ascending=[True, ascending])
            .groupby(["id"])[features]
            .apply(lambda x: pd.Series({"sequence": x.values + shift_value}))
        )
        if df_target is not None:
            chunk = chunk.merge(right=df_target, left_index=True, right_on="id")
        res.append(chunk)
    df = pd.concat(res).reset_index(drop=df_target is not None)
    if presort:
        df = (
            df
            .assign(length=lambda _df: _df["sequence"].apply(len))
            .sort_values(["length"])
            .drop(columns=["length"])
            .reset_index(drop=True)
        )
    return df


def calc_features_agg(df, agg="max"):
    if agg == "min":
        agg_values = np.vstack(df["sequence"].apply(lambda x: x.min(axis=0))).min(axis=0)
    else:
        agg_values = np.vstack(df["sequence"].apply(lambda x: x.max(axis=0))).max(axis=0)
    return agg_values


def pad_sequence(sequences, padding_value=0):
    n_seq = len(sequences)
    max_seq_len = max(seq.shape[0] for seq in sequences)
    residual_dims = sequences[0].shape[1:]
    arr = np.full((n_seq, max_seq_len, *residual_dims), padding_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        arr[i, :seq.shape[0], ...] = seq
    return arr


class PretrainDataset(Dataset):
    def __init__(
        self,
        df,
        vocab_sizes,
        special_tokens_map,
        minlen=6,
        maxlen=30,
        permute_prob=0.5,
        mask_prob=0.15,
        random_replace_prob=0.1,
        keep_unchanged_prob=0.1,
        non_target_idx=-100,
        presort=False,
        random_subseq=False,
        random_permute=False,
    ):
        """
        df: pd.DataFrame с колонками <sequence> <id> (<flag>)
        vocab_sizes: размеры словарей категориальных признаков (с учетом специальных токенов)
        special_tokens_map: dict со значениями индексов специальных токенов
        minlen: минимальная длина выходной последовательности
        maxlen: максимальная длина выходной последовательности
        permute_prob: вероятность, с которой два сегмента меняются местами (происходит swap)
        mask_prob: вероятность выбрать индекс как таргет (сразу по всем признакам)
        random_replace_prob: вероятность для уже выбранного индекса поменять его на случайное значение вместо маскирования
        keep_unchanged_prob: вероятность оставить индекс в исходном виде вместо маскирования
        non_target_idx: значение для индексов, не использующихся как таргеты
        presort: отсортировать датасет по длинам последовательностей (ds[0] - самая короткая)
        random_subseq: если посл-ть больше макс. длины, то выбирается случайная подпосл-ть макс. длины
        random_permute: выполнять ли случайную перестановку сегментов
        """
        super().__init__()
        self._vocab_sizes = np.array(vocab_sizes)
        self._special_tokens_map = special_tokens_map
        self._num_special_tokens = len(self._special_tokens_map)
        self._true_vocab_sizes = self._vocab_sizes - self._num_special_tokens
        self._minlen = minlen
        self.set_maxlen(maxlen=maxlen)
        self._random_subseq = random_subseq
        self._random_permute = random_permute
        self._permute_prob = permute_prob
        self._mask_prob = mask_prob
        self._true_masking_prob = 1.0 - keep_unchanged_prob - random_replace_prob
        self._random_replace_cond_prob = (
            random_replace_prob / (keep_unchanged_prob + random_replace_prob)
        )  # вероятность случайной замены, при условии ~true_masking
        self._non_target_idx = non_target_idx
        df = (
            df
            .assign(length=lambda _df: _df["sequence"].apply(len).astype(np.int8))
            .query(f"length > {self._minlen}")
            .reset_index(drop=True)
        )
        if presort:
            df = (
                df
                .sort_values(["length"])
                .reset_index(drop=True)
            )
        self._df_idx_to_length_mapping = df.assign(idx=lambda _df: _df.index)[["idx", "length"]]
        self._data = df.to_dict(orient="index")
        self._n_features = self._data[0]["sequence"].shape[1]
        self._cls_token_ids = np.full(self._n_features, self._special_tokens_map["cls"], dtype=np.int8)
        self._sep_token_ids = np.full(self._n_features, self._special_tokens_map["sep"], dtype=np.int8)
        self._non_target_token_ids = np.full(self._n_features, self._non_target_idx, dtype=np.int8)

    def generate_buckets(self):
        buckets = (
            self._df_idx_to_length_mapping
            .assign(bucket=lambda _df: _df["length"].clip(upper=self._seq_max_len))
            .groupby("bucket")["idx"].apply(list).to_dict()
        )
        return buckets

    def set_maxlen(self, maxlen, return_buckets=False):
        if self._random_permute:
            self._seq_max_len = maxlen - 2  # + два служебных токена [CLS] и [SEP]
        else:
            self._seq_max_len = maxlen - 1  # + служебный токен [CLS]
        if return_buckets:
            return self.generate_buckets()

    def __len__(self):
        return len(self._data)

    def _permuting(self, seq, labels):
        seq_len = seq.shape[0]
        # разбиваем на два сегмента примерно посередине и с вероятностью _permute_prob меняем их местами
        mid = (seq_len + random.randint(-1, 2)) // 2  # случайность, чтобы не переобучалось
        seq_seg1, seq_seg2 = seq[: mid], seq[mid:]
        lab_seg1, lab_seg2 = labels[: mid], labels[mid:]
        permuted = random.random() < self._permute_prob
        if permuted:
            seq_seg1, seq_seg2 = seq_seg2, seq_seg1
            lab_seg1, lab_seg2 = lab_seg2, lab_seg1
        seq = np.vstack((
            self._cls_token_ids,
            seq_seg1,
            self._sep_token_ids,
            seq_seg2
        ))
        labels = np.vstack((
            self._non_target_token_ids,
            lab_seg1,
            self._non_target_token_ids,
            lab_seg2
        ))
        token_type_ids = np.array(
            [0] * (len(seq_seg1) + 2) + [1] * (len(seq_seg2))
        )
        return seq, labels, permuted, token_type_ids

    def _get_random_subseq(self, seq, copy=False):
        seq_len = seq.shape[0]
        if seq_len > self._seq_max_len:
            start_idx = random.randint(0, seq_len - self._seq_max_len)
            stop_idx = start_idx + self._seq_max_len
            seq = seq[start_idx: stop_idx]
        if copy:
            return seq.copy()
        return seq

    def _masking(self, seq, inplace=False):
        if not inplace:
            seq = seq.copy()
        seq_len = seq.shape[0]
        mask = np.random.rand(seq_len) < self._mask_prob
        # таргеты сохраняем только там, где в маске True
        labels = np.where(mask[:, None], seq, self._non_target_idx)
        # маска для реального маскирования входных индексов токенов
        true_masking_mask = mask & (np.random.rand(seq_len) < self._true_masking_prob)
        # маска для подмены входных индексов токенов
        random_replace_mask = (
            mask & ~true_masking_mask
            & (np.random.rand(seq_len) < self._random_replace_cond_prob)
        )
        # маскируем входы
        seq[true_masking_mask] = self._special_tokens_map["mask"]
        # делаем подмену на входах
        seq[random_replace_mask] = (
            np.random.randint(0, 1000, seq[random_replace_mask].shape)
            % self._true_vocab_sizes + self._num_special_tokens
        )
        return seq, labels

    def __getitem__(self, idx):
        """
        returns:
            input_ids - np.ndarray с индексами,
            labels - np.ndarray со значениями таргетов
            cls - значение cls таргета
        """
        data = self._data[idx]
        seq = data["sequence"]
        if self._random_subseq:
            seq = self._get_random_subseq(seq, copy=False)
        else:
            seq = seq[: self._seq_max_len]
        seq, labels = self._masking(seq, inplace=False)
        if self._random_permute:
            input_ids, labels, permuted, token_type_ids = self._permuting(seq, labels)
            return input_ids, labels, permuted, token_type_ids
        else:
            input_ids = np.vstack((self._cls_token_ids, seq))
            labels = np.vstack((self._non_target_token_ids, labels))
            cls_target = data["flag"]
            return input_ids, labels, cls_target


class PretrainCollator:
    def __init__(
        self,
        special_tokens_map,
        non_target_idx=-100,
        random_permute=False,
    ):
        """
        special_tokens_map: dict со значениями индексов специальных токенов
        non_target_idx: значение для индексов, не использующихся как таргеты
        random_permute: используется ли случайная перестановка (в датасете)
        """
        self._special_tokens_map = special_tokens_map
        self._non_target_idx = non_target_idx
        self._random_permute = random_permute

    def __call__(self, batch):
        if self._random_permute:
            input_ids, labels, cls_target, token_type_ids = zip(*batch)  # cls_target = permuted
            token_type_ids = torch.LongTensor(pad_sequence(token_type_ids, padding_value=1))
        else:
            input_ids, labels, cls_target = zip(*batch)
            token_type_ids = None
        input_ids = torch.LongTensor(pad_sequence(input_ids, padding_value=self._special_tokens_map["pad"]))
        labels = torch.LongTensor(pad_sequence(labels, padding_value=self._non_target_idx))
        cls_target = torch.FloatTensor(cls_target)
        return input_ids, labels, cls_target, token_type_ids


class CommonDataset(Dataset):
    def __init__(self, df, maxlen, presort=False, random_subseq=False):
        """
        df: pd.DataFrame со столбцами <sequence> <id> <flag>
        maxlen: максимальная длина последовательности
        presort: отсортировать последовательности по длине
        random_subseq: если посл-ть больше макс. длины, то выбирается случайная подпосл-ть макс. длины
        """
        super().__init__()
        self._maxlen = maxlen
        self._random_subseq = random_subseq
        if presort:
            df = (
                df
                .assign(length=lambda _df: _df["sequence"].apply(len))
                .sort_values(["length"])
                .drop(columns=["length"])
                .reset_index(drop=True)
            )
        self._df_idx_to_length_mapping = (
            df
            .assign(idx=lambda _df: _df.index)
            .assign(length=lambda _df: _df["sequence"].apply(len))[["idx", "length"]]
        )
        self._data = df.to_dict(orient="index")

    def __len__(self):
        return len(self._data)

    def generate_buckets(self):
        buckets = (
            self._df_idx_to_length_mapping
            .assign(bucket=lambda _df: _df["length"].clip(upper=self._maxlen))
            .groupby("bucket")["idx"].apply(list).to_dict()
        )
        return buckets

    def set_maxlen(self, maxlen, return_buckets=False):
        self._maxlen = maxlen
        if return_buckets:
            return self.generate_buckets()

    def _get_random_subseq(self, seq, copy=False):
        seq_len = seq.shape[0]
        if seq_len > self._maxlen:
            start_idx = random.randint(0, seq_len - self._maxlen)
            stop_idx = start_idx + self._maxlen
            seq = seq[start_idx: stop_idx]
        if copy:
            return seq.copy()
        return seq

    def __getitem__(self, idx):
        """
        returns:
            input_ids - nd.array с индексами токенов последовательности
            target - целевая переменная (0 или 1)
            uid - уникальный id объекта
        """
        data = self._data[idx]
        if self._random_subseq:
            input_ids = self._get_random_subseq(data["sequence"])
        else:
            input_ids = data["sequence"][: self._maxlen, :]
        target = data["flag"]
        uid = data["id"]
        return input_ids, target, uid


class CommonCollator:
    def __init__(
        self,
        padding_value=0,
        reverse=False,
        dropout_prob=0,
    ):
        """
        padding_value: значение паддинга
        reverse: отражать ли зеркально полседовательности
        dropout_prob: вероятность зануления элемента последовательности
        """
        self._padding_value = padding_value
        self._reverse = reverse
        self._dropout_prob = dropout_prob

    def __call__(self, batch):
        input_ids, targets, uids = zip(*batch)
        if self._reverse:
            input_ids = [np.flip(x, axis=0) for x in input_ids]
        input_ids = pad_sequence(input_ids, padding_value=self._padding_value)
        if self._dropout_prob > 0 and input_ids.shape[1] > 1:
            mask = np.random.rand(*input_ids.shape) > self._dropout_prob
            input_ids = input_ids * mask + self._padding_value * ~mask
        input_ids = torch.LongTensor(input_ids)
        targets = torch.FloatTensor(targets)
        uids = torch.LongTensor(uids)
        return input_ids, targets, uids


class BucketSampler(Sampler):
    def __init__(self, buckets, shuffle=False):
        self._buckets = buckets
        self._shuffle = shuffle
        self._length = sum(len(idxs) for idxs in self._buckets.values())

    def __iter__(self):
        if self._shuffle:
            for bucket in self._buckets:
                np.random.shuffle(self._buckets[bucket])
        for bucket in sorted(self._buckets.keys()):
            yield from self._buckets[bucket]

    def __len__(self):
        return self._length


class InferenceDataset(Dataset):
    def __init__(self, df, maxlen, presort=False):
        """
        df: pd.DataFrame со столбцами <sequence> <id>
        maxlen: максимальная длина последовательности
        presort: отсортировать последовательности по длине
        """
        super().__init__()
        self._maxlen = maxlen
        if presort:
            df = (
                df
                .assign(length=lambda _df: _df["sequence"].apply(len))
                .sort_values(["length"])
                .drop(columns=["length"])
                .reset_index(drop=True)
            )
        self._data = df.to_dict(orient="index")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """
        returns:
            uid - уникальный идентификатор объекта
            input_ids - nd.array с индексами токенов последовательности
        """
        data = self._data[idx]
        uid = data["id"]
        input_ids = data["sequence"][: self._maxlen, :]
        return uid, input_ids


class InferenceCollator:
    def __init__(self, padding_value=0, reverse=False, vocab_sizes=None):
        self._padding_value = padding_value
        self._reverse = reverse
        if vocab_sizes is not None:
            self._max_ids = vocab_sizes[None, None, ...] - 1  # 1 x 1 x F
        else:
            self._max_ids = None

    def __call__(self, batch):
        uid, input_ids = zip(*batch)
        if self._reverse:
            input_ids = [np.flip(x, axis=0) for x in input_ids]
        input_ids = pad_sequence(input_ids, padding_value=self._padding_value)
        if self._max_ids is not None:
            input_ids[input_ids > self._max_ids] = self._padding_value
        input_ids = torch.LongTensor(input_ids)
        uid = torch.LongTensor(uid)
        return uid, input_ids
