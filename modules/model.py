import math
import torch
import torch.nn.functional as F
from itertools import accumulate
from torch import nn


def init_layer(layer, initializer_range=0.02, zero_out_bias=True):
    """
    layer: наследник nn.Module, т.е. слой в pytorch
    initializer_range: stddev для truncated normal
    zero_out_bias: True для линейных слоев, False для матрицы эмбеддингов
    """
    stddev = initializer_range
    nn.init.trunc_normal_(layer.weight.data, std=stddev, a=-2*stddev, b=2*stddev)
    if zero_out_bias:
        layer.bias.data.zero_()


def load_weights(model, path):
    found = []
    with open(path, 'rb') as f:
        weights = torch.load(f)
        print(f'Total amount weights: {len(weights)}')
    for name, param in weights.items():
        if name in model.state_dict():
            if param.shape == model.state_dict()[name].shape:
                model.state_dict()[name].copy_(param)
                found.append(name)
    return found


class LabelSmoothingBCEWithLogitsLoss:
    def __init__(
        self,
        weighted=False,
        label_smoothing=True,
        alpha=0.1,
        period=None,
        pos_weight=None,
    ):
        """
        weighted: использовать ли веса для объектов при вычислении функции потерь
        label_smoothing: использовать ли label_smoothing
        alpha: предельное значение искажения таргета для label smoothing
        period: период, определяет насколько медленно уменьшается label smoothing
            для объектов с более высокими uid (чем больше, тем медленнее уменьшение);
            Закон изменения - экспоненциальный, для uid = 0 таргет изменяется на alpha.
            Если None, то таргет изменяется на alpha для всех uid.
        pos_weight: коэффициент, на который будет умножаться лосс объектов положительного класса
        """
        self._weighted = weighted
        self._label_smoothing = label_smoothing
        self._alpha = alpha
        self._period = period
        if self._weighted:
            self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        else:
            self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def __call__(self, input, target, uid=None):
        if self._weighted:
            if uid is not None:
                weights = self._alpha * torch.exp(-uid / self._period)
                return torch.sum(self._criterion(input, target) * weights) / torch.sum(weights)
            else:
                return torch.mean(self._criterion(input, target))
        else:
            if self._label_smoothing:
                alpha = self._alpha
                if self._period is not None and uid is not None:
                    alpha = alpha * torch.exp(-uid / self._period)
                target = (1 - alpha) * target + alpha / 2
            return self._criterion(input, target)


class FullyConnectedMLP(nn.Module):
    def __init__(self, input_dim, hiddens, output_dim, dropout_prob=0., act_func='relu'):
        """
        input_dim: входная размерность
        hiddens: список внутренних размерностей
        output_dim: выходная размерность
        dropout_prob: вероятность дропаута
        act_func: функция активации
        """
        assert isinstance(hiddens, list)
        super().__init__()
        act_func_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
        }
        layers = []
        dims = [input_dim] + hiddens + [output_dim]
        for in_features, out_features in zip(dims[:-2], dims[1:]):
            layers += [
                nn.Linear(in_features, out_features),
                act_func_map[act_func](),
                nn.Dropout(dropout_prob)
            ]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self._net = nn.Sequential(*layers)
        # self.init_weights()

    def init_weights(self):
        for layer in self._net:
            if isinstance(layer, nn.Linear):
                init_layer(layer)

    def forward(self, x):
        return self._net(x)


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_layer_sizes,
        max_seqlen,
        dropout_prob=0.,
        type_vocab_size=2,
        eps=1e-3,
        padding_idx=None,
    ):
        """
        embedding_layer_sizes: список размеров слоев эмбеддингов в виде пар
            (vocab_size, embedding_size)
        max_seqlen: количество позиционных эмбеддингов
        dropout_prob: вероятность дропаута в конце слоя
        type_vocab_size: количество сегментных эмбеддингов
        eps: eps для layernorm
        padding_idx: индекс паддинга для эмбеддингов, чтобы паддинг был необучаемым
        """
        super().__init__()
        self._token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
            for vocab_size, embedding_size in embedding_layer_sizes
        ])
        self._hidden_size = sum([embedding_size for vocab_size, embedding_size in embedding_layer_sizes])
        self._pos_embeddings = nn.Embedding(max_seqlen, self._hidden_size)
        self._seg_embeddings = nn.Embedding(type_vocab_size, self._hidden_size)
        self._layernorm = nn.LayerNorm(self._hidden_size, eps=eps)
        self._dropout = nn.Dropout(dropout_prob)
        self.init_weights()

    def init_weights(self):
        for embeddings in self._token_embeddings:
            init_layer(embeddings, initializer_range=0.02, zero_out_bias=False)
        init_layer(self._pos_embeddings, initializer_range=0.02, zero_out_bias=False)
        init_layer(self._seg_embeddings, initializer_range=0.02, zero_out_bias=False)

    def get_token_embeddings(self, i):
        """
        returns: возвращает слой с матрицей эмбеддингов для токенов. Нужен для MLM головы
        """
        return self._token_embeddings[i]

    def forward(self, input_ids, token_type_ids=None):
        """
        input_ids: тензор с индексами токенов
        token_type_ids: сегментные индексы

        returns: эмбеддинги токенов
        """
        B, S, F = input_ids.shape
        pos = torch.arange(S).repeat(B, 1).to(input_ids.device)
        embs = torch.cat([self._token_embeddings[i](input_ids[:, :, i]) for i in range(F)], dim=-1)  # [B x S x D]
        embs = embs + self._pos_embeddings(pos)
        if token_type_ids is not None:
            embs = embs + self._seg_embeddings(token_type_ids)
        return self._dropout(self._layernorm(embs))


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob=0.,
        dropout_prob=0.,
        eps=1e-3
    ):
        """
        hidden_size: размерность эмбеддингов
        num_attention_heads: количество голов аттеншна. Обычно выбирается как hidden_size / num_attention_heads = 64,
            т.е. размерность векторов у одной головы 64
        attention_probs_dropout_prob: вероятность дропаута для аттеншн скоров
        dropout_prob: вероятность дропаута в конце слоя (перед суммой со входами)
        eps: eps для layernorm
        """
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self._hidden_size = hidden_size
        self._num_heads = num_attention_heads
        self._linear = nn.Linear(hidden_size, 3 * hidden_size)
        self._att_dropout = nn.Dropout(attention_probs_dropout_prob)
        self._out_linear = nn.Linear(hidden_size, hidden_size)
        self._out_dropout = nn.Dropout(dropout_prob)
        self._layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self.init_weights()

    def init_weights(self):
        init_layer(self._linear)
        init_layer(self._out_linear)

    @property
    def size_per_head(self):
        """
        returns: размерность векторов для одной головы
        """
        return self._hidden_size // self._num_heads

    def forward(self, embeddings, attention_mask):
        """
        embeddings: входные эмбеддинги
        attention_mask: тензор из 0, 1 размерности batch_size x seqlen x seqlen

        returns: контекстные векторы
        """
        B, S, D = embeddings.shape
        H = self._num_heads
        A = self.size_per_head

        query, key, value = self._linear(embeddings).chunk(3, dim=-1)  # [B x S x D] x 3

        query = query.view(B, S, H, A).transpose(1, 2)  # [B x H x S x A]
        key = key.view(B, S, H, A).transpose(1, 2)
        value = value.view(B, S, H, A).transpose(1, 2)

        att_mask = attention_mask.unsqueeze(1)  # [B x 1 x S x S]

        att_scores = torch.matmul(query, key.transpose(-2, -1))  # [B x H x S x S]
        att_scores = att_mask * att_scores + (1 - att_mask) * -100000
        att_scores = att_scores / math.sqrt(A)

        att_scores = self._att_dropout(att_scores)
        att_scores = torch.nn.functional.softmax(att_scores, dim=-1)

        x = torch.matmul(att_scores, value)  # [B x H x S x A]
        x = x.transpose(1, 2).contiguous().view(B, S, -1)  # [B x S x D]
        x = self._layernorm(embeddings + self._out_dropout(self._out_linear(x)))
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        dropout_prob=0.,
        act_func='relu',
        eps=1e-3
    ):
        """
        hidden_size: размерность эмбеддингов
        intermediate_size: размерность промежуточно слоя. Обычно 4 * hidden_size
        dropout_prob: вероятность дропаута перед суммой со входными представлениями
        act_func: функция активации. Должны быть доступны gelu, relu
        eps: eps для layernorm
        """
        super().__init__()
        self._linear1 = nn.Linear(hidden_size, intermediate_size)
        self._linear2 = nn.Linear(intermediate_size, hidden_size)
        self._dropout = nn.Dropout(dropout_prob)
        self._layernorm = nn.LayerNorm(hidden_size, eps=eps)
        if act_func == 'gelu':
            self._activation = nn.GELU()
        else:
            self._activation = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        init_layer(self._linear1)
        init_layer(self._linear2)

    def forward(self, embeddings):
        """
        embeddings: входные эмбеддинги размерности batch_size x seqlen x hidden_size

        returns: преобразованные эмбеддинги той же размерности
        """
        x = self._activation(self._linear1(embeddings))
        x = self._dropout(self._linear2(x))
        x = self._layernorm(x + embeddings)
        return x


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        act_func='relu',
        eps=1e-3
    ):
        super().__init__()
        self._multihead_attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            dropout_prob=dropout_prob,
            eps=eps
        )
        self._feedforward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            act_func=act_func,
            eps=eps,
            dropout_prob=dropout_prob
        )

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            B, S, D = x.shape
            attention_mask = torch.ones(B, S, S, device=x.device)
        x = self._multihead_attention(x, attention_mask)
        x = self._feedforward(x)
        return x


class Bert(nn.Module):
    def __init__(
        self,
        embedding_layer_sizes,
        max_seqlen,
        num_hidden_layers,
        intermediate_size,
        num_attention_heads,
        input_dropout_prob=0.,
        dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        act_func='relu',
        eps=1e-3,
        padding_idx=None,
    ):
        super().__init__()
        self._embeddings = BertEmbeddings(
            embedding_layer_sizes,
            max_seqlen,
            input_dropout_prob,
            type_vocab_size=2,
            eps=eps,
            padding_idx=padding_idx
        )
        self._bert_layers = nn.ModuleList(
            BertLayer(
                self._embeddings._hidden_size,
                intermediate_size,
                num_attention_heads,
                dropout_prob,
                attention_probs_dropout_prob,
                act_func,
                eps=eps
            ) for _ in range(num_hidden_layers)
        )

    def get_token_embeddings(self, i):
        """
        returns: эмбеддинги токенов (матрицу эмбеддингов)
        """
        return self._embeddings.get_token_embeddings(i)

    @staticmethod
    def expand_mask(attention_mask):
        """
        attention_mask: маска паддинга размерности batch_size x seqlen

        returns: маска паддинга размерности batch_size x seqlen x seqlen
        """
        # return attention_mask[:, None] * attention_mask[..., None]
        return attention_mask[:, None] * torch.ones_like(attention_mask)[..., None]

    def forward(self, x, attention_mask=None, token_type_ids=None):
        if attention_mask is not None:
            attention_mask = self.expand_mask(attention_mask)
        x = self._embeddings(x, token_type_ids)
        for bert_layer in self._bert_layers:
            x = bert_layer(x, attention_mask)
        return x


class MlmHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        hidden_act,
        eps=1e-3,
        ignore_index=-100,
        input_embeddings=None
    ):
        """
        hidden_size: размерность эмбеддингов
        vocab_size: размер словаря
        hidden_act: функция активации
        eps: eps для layernorm
        ignore_index: индекс таргета, который необходимо игнорировать при подсчете лосса
        input_embeddings: слой с эмбеддингами токенов, для использования матрицы эмбеддингов вместо линейного слоя
        """
        super().__init__()
        self._vocab_size = vocab_size
        self._linear1 = nn.Linear(hidden_size, hidden_size)
        if hidden_act == 'gelu':
            self._activation = nn.GELU()
        else:
            self._activation = nn.ReLU()
        self._layernorm = nn.LayerNorm(hidden_size, eps=eps)
        self._linear2 = nn.Linear(hidden_size, vocab_size)
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.init_weights(input_embeddings=input_embeddings)

    def init_weights(self, input_embeddings=None):
        init_layer(self._linear1)
        if input_embeddings is not None:
            self._linear2.weight = input_embeddings.weight
        else:
            init_layer(self._linear2)

    def forward(self, hidden_states, labels):
        """
        hidden_states: эмбеддинги токенов
        labels: истинные метки, т.е. изначальные индексы токенов

        returns: посчитанный лосс
        """
        preds = self._linear2(self._layernorm(self._activation(self._linear1(hidden_states))))
        loss = self._criterion(preds.view(-1, self._vocab_size), labels.view(-1))
        return loss


class ClassifierHead(nn.Module):
    CLS_POSITION = 0
    CRITERION = nn.BCEWithLogitsLoss()

    def __init__(self, hidden_size, num_classes=1, hidden_dropout_prob=0.):
        """
        hidden_size: размерность эмбеддингов
        hidden_dropout_prob: вероятность дропаута
        """
        super().__init__()
        self._linear1 = nn.Linear(hidden_size, hidden_size)
        self._activation = nn.Tanh()
        self._dropout = nn.Dropout(hidden_dropout_prob)
        self._linear2 = nn.Linear(hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        init_layer(self._linear1)
        init_layer(self._linear2)

    def forward(self, hidden_states, permuted=None):
        """
        hidden_states: эмбеддинги
        permuted: таргеты (были ли свапы сегментов). Если их нет, то необходимо выдать предсказания
        """
        x = hidden_states[:, self.CLS_POSITION, :]
        x = self._dropout(self._activation(self._linear1(x)))
        logits = self._linear2(x)
        if permuted is not None:
            loss = self.CRITERION(logits.squeeze(), permuted)
            return loss
        else:
            return logits


class PoolingClassifierHead(nn.Module):
    def __init__(self, input_dim, output_dim, hiddens=None, dropout_prob=0., act_func='relu'):
        """
        input_dim: входная размерность
        hiddens: список внутренних размерностей
        output_dim: выходная размерность
        dropout_prob: вероятность дропаута
        act_func: функция активации
        """
        super().__init__()
        if hiddens is None:
            hiddens = [2 * input_dim, input_dim]
        self._fc = FullyConnectedMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hiddens=hiddens,
            dropout_prob=dropout_prob,
            act_func=act_func,
        )

    def forward(self, hidden_states):
        """
        hidden_states: эмбеддинги (batch_size x seqlen x hidden_size)
        """
        seq_max_pool = torch.max(hidden_states, dim=1)[0]
        seq_avg_pool = torch.sum(hidden_states, dim=1) / hidden_states.shape[1]
        seq_state = torch.cat([seq_max_pool, seq_avg_pool], dim=-1)
        logits = self._fc(seq_state)
        return logits


class ConvClassifierHead(nn.Module):
    def __init__(self, embedding_size, n_features=128, window_sizes=(1, 3, 5), fc_dropout_prob=0.):
        """
        embedding_size: размерность входных эмбеддингов
        n_features: количество выходных каналов для сверток
        window_sizes: размеры окон (высоты сверток)
        fc_dropout_prob: вероятность дропаута в линейном слое
        """
        super().__init__()
        self._convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, n_features, kernel_size=(window_size, embedding_size), padding=(window_size // 2, 0)),
                nn.BatchNorm2d(n_features),
                nn.ReLU(),
            )
            for window_size in window_sizes
        ])
        self._fc = FullyConnectedMLP(
            input_dim=len(window_sizes) * n_features,
            hiddens=[n_features],
            output_dim=1,
            dropout_prob=fc_dropout_prob,
            act_func='relu'
        )

    def forward(self, hidden_states):
        """
        hidden_states: эмбеддинги (batch_size x seqlen x hidden_size)
        """
        x = hidden_states.unsqueeze(1)  # [N, 1, S, D]
        xs = []
        for conv in self._convs:
            x2 = conv(x)  # [N, C, S, 1]
            x2 = F.max_pool2d(x2, kernel_size=(x2.shape[-2], 1))  # [N, C, 1, 1]
            xs.append(x2.view(x2.shape[:2]))
        x = torch.cat(xs, dim=1)  # [N, C * len(window_sizes)]
        logits = self._fc(x)
        return logits


class BertModel(nn.Module):
    def __init__(
        self,
        embedding_layer_sizes,
        max_seqlen,
        num_hidden_layers,
        intermediate_size,
        num_attention_heads,
        act_func='relu',
        input_dropout_prob=0.,
        hidden_dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        eps=1e-3,
        ignore_index=-100,
        mlm_loss_weights=None,
        padding_idx=None,
    ):
        super().__init__()
        embedding_sizes = list(zip(*embedding_layer_sizes))[1]
        self._num_embeddings = len(embedding_sizes)
        self._hidden_size = sum(embedding_sizes)
        self._embedding_start_idxs = [0] + list(accumulate(embedding_sizes))
        if mlm_loss_weights is None:
            self._mlm_loss_weights = [1] * self._num_embeddings
        else:
            self._mlm_loss_weights = mlm_loss_weights
        self._backbone = Bert(
            embedding_layer_sizes=embedding_layer_sizes,
            max_seqlen=max_seqlen,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            input_dropout_prob=input_dropout_prob,
            dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            act_func=act_func,
            eps=eps,
            padding_idx=padding_idx
        )
        self._mlm_heads = nn.ModuleList([
            MlmHead(
                hidden_size=embedding_layer_size[1],
                vocab_size=embedding_layer_size[0],
                hidden_act=act_func,
                eps=eps,
                ignore_index=ignore_index,
                input_embeddings=self._backbone.get_token_embeddings(i)
            ) for i, embedding_layer_size in enumerate(embedding_layer_sizes)
        ])
        self._classifier_head = ClassifierHead(
            self._hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            num_classes=1
        )

    def forward(self, x, attention_mask, labels, permuted, token_type_ids=None):
        hidden_states = self._backbone(x, attention_mask, token_type_ids)
        mlm_losses = []
        for i in range(self._num_embeddings):
            start_idx = self._embedding_start_idxs[i]
            stop_idx = self._embedding_start_idxs[i + 1]
            mlm_losses.append(self._mlm_heads[i](hidden_states[:, :, start_idx: stop_idx], labels[:, :, i]))
        mlm_loss = sum(loss * weight for loss, weight in zip(mlm_losses, self._mlm_loss_weights)) / sum(self._mlm_loss_weights)
        sop_loss = self._classifier_head(hidden_states, permuted)
        # в оригинальном BERT лоссы MLP и NSP используются с равными весами
        return 0.5 * mlm_loss + 0.5 * sop_loss, {'MLM': mlm_loss, 'SOP': sop_loss}


class BertFinetuneModel(nn.Module):
    def __init__(
        self,
        embedding_layer_sizes,
        max_seqlen,
        num_hidden_layers,
        intermediate_size,
        num_attention_heads,
        act_func='relu',
        input_dropout_prob=0.,
        hidden_dropout_prob=0.,
        attention_probs_dropout_prob=0.,
        eps=1e-3,
        clf_head_hiddens=None,
        clf_head_dropout_prob=0.,
        padding_idx=None,
        clf_head_type='pooling',
    ):
        super().__init__()
        embedding_sizes = list(zip(*embedding_layer_sizes))[1]
        self._hidden_size = sum(embedding_sizes)
        self._backbone = Bert(
            embedding_layer_sizes=embedding_layer_sizes,
            max_seqlen=max_seqlen,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            input_dropout_prob=input_dropout_prob,
            dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            act_func=act_func,
            eps=eps,
            padding_idx=padding_idx
        )
        if clf_head_type == 'classic':
            self._classifier_head = ClassifierHead(self._hidden_size, 1, clf_head_dropout_prob)
        elif clf_head_type == 'pooling':
            self._classifier_head = PoolingClassifierHead(
                input_dim=2 * self._hidden_size,
                output_dim=1,
                hiddens=clf_head_hiddens,
                dropout_prob=clf_head_dropout_prob,
                act_func='relu',
            )
        else:  # 'conv'
            self._classifier_head = ConvClassifierHead(
                embedding_size=self._hidden_size,
                n_features=128,
                window_sizes=(1, 3, 5),
                fc_dropout_prob=clf_head_dropout_prob,
            )

    def forward(self, x, attention_mask):
        hidden_states = self._backbone(x, attention_mask)  # batch_size x seqlen x hidden_size
        return self._classifier_head(hidden_states)
