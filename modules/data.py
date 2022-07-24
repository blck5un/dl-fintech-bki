import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
    df = pd.concat(res)
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
        """
        super().__init__()
        self._vocab_sizes = np.array(vocab_sizes)
        self._special_tokens_map = special_tokens_map
        self._num_special_tokens = len(self._special_tokens_map)
        self._true_vocab_sizes = self._vocab_sizes - self._num_special_tokens
        self._minlen = minlen
        self._maxlen = maxlen - 2  # + два служебных токена [CLS] и [SEP]
        self._permute_prob = permute_prob
        self._mask_prob = mask_prob
        self._true_masking_prob = 1.0 - keep_unchanged_prob - random_replace_prob
        self._random_replace_cond_prob = (
            random_replace_prob / (keep_unchanged_prob + random_replace_prob)
        )  # вероятность случайной замены, при условии ~true_masking
        self._non_target_idx = non_target_idx
        df = (
            df
            .assign(length=lambda _df: _df["sequence"].apply(len))
            .query(f"length > {self._minlen}")
            .drop(columns=["length"])
            .reset_index(drop=True)
        )
        if presort:
            df = (
                df
                .sort_values(["length"])
                .drop(columns=["length"])
                .reset_index(drop=True)
            )
        self._data = df.to_dict(orient="index")
        self._n_features = self._data[0]["sequence"].shape[1]
        self._cls_token_ids = np.full(self._n_features, self._special_tokens_map["cls"], dtype=np.int8)
        self._sep_token_ids = np.full(self._n_features, self._special_tokens_map["sep"], dtype=np.int8)
        self._non_target_token_ids = np.full(self._n_features, self._non_target_idx, dtype=np.int8)

    def set_maxlen(self, maxlen):
        """
        Устанавливает новое максимальное значение длины выходной последовательности
        maxlen: максимальная длина выходной последовательности
        """
        self._maxlen = maxlen - 2  # + два служебных токена [CLS] и [SEP]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        """
        returns:
            input_ids - np.ndarray с индексами,
            token_type_ids - np.ndarray с сегментными айдишниками (0 у левого сегмента, 1 у правого),
            labels - np.ndarray со значениями таргетов
            permuted - bool был ли swap сегментов
        """
        data = self._data[idx]
        seq_len = data["sequence"].shape[0]
        if seq_len > self._maxlen:
            start_idx = random.randint(0, seq_len - self._maxlen)
            stop_idx = start_idx + self._maxlen
            seq = data["sequence"][start_idx: stop_idx].copy()
        else:
            seq = data["sequence"].copy()
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
            % self._true_vocab_sizes
            + self._num_special_tokens
        )
        # разбиваем на два сегмента примерно по середине и с вероятностью _permute_prob меняем их местами
        mid = (seq_len + random.randint(-1, 2)) // 2  # случайность, чтобы не переобучалось
        seq_seg1, seq_seg2 = seq[: mid], seq[mid:]
        lab_seg1, lab_seg2 = labels[: mid], labels[mid:]
        permuted = random.random() < self._permute_prob
        if permuted:
            seq_seg1, seq_seg2 = seq_seg2, seq_seg1
            lab_seg1, lab_seg2 = lab_seg2, lab_seg1
        input_ids = np.vstack((
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
            [0] * (len(seq_seg1) + 2)
            + [1] * (len(seq_seg2))
        )
        return input_ids, token_type_ids, labels, permuted


class PretrainCollator:
    def __init__(
        self,
        special_tokens_map,
        non_target_idx=-100,
    ):
        """
        special_tokens_map: dict со значениями индексов специальных токенов
        non_target_idx: значение для индексов, не использующихся как таргеты
        """
        self._special_tokens_map = special_tokens_map
        self._non_target_idx = non_target_idx

    def __call__(self, batch):
        input_ids, token_type_ids, labels, permuted = zip(*batch)
        input_ids = torch.LongTensor(pad_sequence(input_ids, padding_value=self._special_tokens_map["pad"]))
        token_type_ids = torch.LongTensor(pad_sequence(token_type_ids, padding_value=1))
        labels = torch.LongTensor(pad_sequence(labels, padding_value=self._non_target_idx))
        permuted = torch.FloatTensor(permuted)
        return input_ids, token_type_ids, labels, permuted


class CommonDataset(Dataset):
    def __init__(self, df, maxlen, presort=False, inference=False):
        """
        df: pd.DataFrame со столбцами <sequence> <id> <flag>
            (стобей <flag> не требуется, если inference=True)
        maxlen: максимальная длина последовательности
        presort: отсортировать последовательности по длине
        inference: режим использования датасета:
            - True для инференса
            - False для трейна
        """
        super().__init__()
        self._maxlen = maxlen
        self._inference = inference
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

    def set_maxlen(self, maxlen):
        self._maxlen = maxlen

    def __getitem__(self, idx):
        """
        returns:
            input_ids - nd.array с индексами токенов последовательности
            target - целевая переменная (0 или 1)
        """
        data = self._data[idx]
        input_ids = data["sequence"][: self._maxlen, :]
        if self._inference:
            return input_ids
        target = data["flag"]
        return input_ids, target


class CommonCollator:
    def __init__(self, padding_value=0, reverse=False, dropout_prob=0, inference=False):
        self._padding_value = padding_value
        self._reverse = reverse
        self._dropout_prob = dropout_prob
        self._inference = inference

    def __call__(self, batch):
        if self._inference:
            input_ids = batch
        else:
            input_ids, targets = zip(*batch)
            targets = torch.FloatTensor(targets)
        if self._reverse:
            input_ids = [np.flip(x, axis=0) for x in input_ids]
        input_ids = pad_sequence(input_ids, padding_value=self._padding_value)
        if self._dropout_prob > 0:
            mask = np.random.rand(*input_ids.shape) > self._dropout_prob
            input_ids = input_ids * mask
        input_ids = torch.LongTensor(input_ids)
        if self._inference:
            return input_ids
        return input_ids, targets
