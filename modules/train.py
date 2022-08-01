import time
import tqdm
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from model import LabelSmoothingBCEWithLogitsLoss


def get_optimizer(model, weight_decay=0.01):
    """
    model: инициализированная модель
    weight_decay: коэффициент l2 регуляризации

    returns: оптимизатор
    """
    decayed_parameters, not_decayed_parameters = [], []
    for param_name, param in model.named_parameters():
        if param_name.find('layernorm') != -1 or param_name.find('bias') != -1:
            not_decayed_parameters.append(param)
        else:
            decayed_parameters.append(param)
    grouped_parameters = [
        {'params': decayed_parameters, 'weight_decay': weight_decay},
        {'params': not_decayed_parameters, 'weight_decay': 0.}
    ]
    return torch.optim.AdamW(grouped_parameters)


def roc_auc_scorer(target, pred):
    return "roc_auc", roc_auc_score(target, pred)


def inference(model, dataloader, device):
    """
    model: модель, с помощью которой будет осуществляться инференс
    dataloader: даталоадер тестового датасета
    device: девайс, на который будут перемещаться данные
    returns: pd.DataFrame со столбцами <id> <score>
    """
    model.eval()
    uid_list = []
    preds_list = []
    with torch.no_grad():
        for uid, input_ids in dataloader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            uid_list.append(uid)
            preds_list.append(logits.squeeze(-1).cpu())
    uids = torch.cat(uid_list).numpy()
    preds = torch.cat(preds_list).numpy()
    df_preds = pd.DataFrame({"id": uids, "score": preds})
    return df_preds


class Scheduler:
    def __init__(
        self,
        optimizer,
        init_lr,
        peak_lr,
        final_lr,
        num_warmup_steps,
        num_training_steps
    ):
        """
        optimizer: оптимизатор
        init_lr: начальное значение learning rate
        peak_lr: пиковое значение learning rate
        final_lr: финальное значение lr
        num_warmup_steps: количество шагов разогрева (сколько шагов идем от начального до пикового значения)
        num_training_steps: количество шагов обучения (количество батчей x количество эпох)
        """
        self._optimizer = optimizer
        self._step = 0
        self._lr_list = (
            np.linspace(init_lr, peak_lr, num_warmup_steps).tolist() +
            np.linspace(peak_lr, final_lr, num_training_steps - num_warmup_steps).tolist()
        )
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr_list[self._step]

    def step(self):
        """
        Меняет learning rate для оптимизатора
        Поменять learning rate для группы параметров в оптимизаторе можно присваиванием вида param_group['lr'] = lr
        """
        self._step += 1
        if self._step < len(self._lr_list):
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = self._lr_list[self._step]

    def get_last_lr(self):
        """
        returns: текущий learning rate оптимизатора. Нужно для логгирования
        """
        return [param_group['lr'] for param_group in self._optimizer.param_groups]


class PreTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        num_accum_steps=1,
        logdir=None,
        max_grad_norm=None,
    ):
        """
        model: объект класса BertModel
        optimizer: оптимизатор
        scheduler: расписание learning rate. Нужно вызывать scheduler.step() ПОСЛЕ optimizer.step()
        device: девайс (cpu или cuda), на котором надо производить вычисления
        num_accum_steps: количество шагов аккумуляции градиента
        logdir: директория для записи логов
        max_grad_norm: максимум нормы градиентов, для клиппинга
        """
        self._model = model
        self._optimizer = optimizer
        # self._model, self._optimizer = amp.initialize(model, optimizer, opt_level='O1')
        self._scheduler = scheduler
        self._device = device
        self._num_accum_steps = num_accum_steps
        self._logdir = logdir
        self._max_grad_norm = max_grad_norm

    def train(self, dataloader, n_epochs):
        self._writer = SummaryWriter(self._logdir)
        self._model.train()
        self._n_iter = 0
        for epoch in range(1, n_epochs + 1):
            self._train_step(dataloader, epoch)
        self._writer.close()

    def _train_step(self, dataloader, epoch):
        """
        dataloader: объект класса DataLoader для обучения
        epoch: номер эпохи обучения
        """
        self._optimizer.zero_grad()
        for input_ids, labels, cls_target, token_type_ids in tqdm.tqdm(dataloader, desc=f"Epoch: {epoch:02}", mininterval=2):
            input_ids = input_ids.to(self._device)
            labels = labels.to(self._device)
            cls_target = cls_target.to(self._device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self._device)
            loss, dct_losses = self._model(input_ids, labels, cls_target, token_type_ids)
            current_lr = self._scheduler.get_last_lr()
            # logging
            self._writer.add_scalar('Pre-train/total_loss', loss.item(), self._n_iter)
            for loss_name, loss_value in dct_losses.items():
                self._writer.add_scalar(f'Pre-train/{loss_name}_loss', loss_value.item(), self._n_iter)
            self._writer.add_scalars(
                'Pre-train/lr',
                {f'group[{group_id}]': lr for group_id, lr in enumerate(current_lr)},
                self._n_iter
            )
            # backprop
            loss = loss / self._num_accum_steps
            # with amp.scale_loss(loss, self._optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            # step
            self._n_iter += 1
            if self._n_iter % self._num_accum_steps == 0:
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._optimizer.step()
                self._scheduler.step()
                self._optimizer.zero_grad()


class FinetuneTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        logdir=None,
        max_grad_norm=None,
        num_accum_steps=1,
        checkpoint_path="best_checkpoint.pt",
        weighted_loss=False,
        scheduler_step_after_epoch=False,
    ):
        """
        model: объект класса BertModel
        optimizer: оптимизатор
        scheduler: расписание learning rate. Нужно вызывать scheduler.step() ПОСЛЕ optimizer.step()
        criterion: функция потерь
        device: девайс (cpu или cuda), на котором надо производить вычисления
        logdir: директория для записи логов
        max_grad_norm: максимум нормы градиентов, для клиппинга
        num_accum_steps: количество шагов аккумуляции градиента
        checkpoint_path: путь для сохранения лучшей модели
        weighted_loss: использовать ли для каждого объекта свой вес при подсчете лосса
        scheduler_step_after_epoch: выполнять шаг планировщика в конце каждой эпохи
            (иначе после каждого шага оптимизатора)
        """
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._criterion = criterion
        self._device = device
        self._logdir = logdir
        self._max_grad_norm = max_grad_norm
        self._num_accum_steps = num_accum_steps
        self._checkpoint_path = checkpoint_path
        self._verbose = True
        self._label_smoothing = isinstance(self._criterion, LabelSmoothingBCEWithLogitsLoss)
        self._weighted_loss = weighted_loss
        self._scheduler_step_after_epoch = scheduler_step_after_epoch

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def train(
        self,
        dataloaders,
        n_epochs,
        scorer,
        save_checkpoint=True,
        score_th=0.5,
        verbose=True,
        break_after_epoch=None,
    ):
        """
        dataloaders: dict of dataloaders, keys 'train', 'valid' should be present.
        n_epochs: int. Num epochs to train for.
        scorer: takes dataloader, outputs metric name and value as a tuple.
        save_checkpoint: сохранять ли checkpoint модели при улучшении скора
        score_th: минимальный порог скора для сохранения checkpoint
        verbose: выводить ли информацию о процессе обучения
        break_after_epoch: прервать обучение после заданной эпохи
        """
        result_list = []
        self._best_score = 0
        self._n_iter_train = 0
        self._writer = SummaryWriter(self._logdir)
        self._verbose = verbose
        epoch_iter = range(1, n_epochs + 1) if self._verbose else tqdm.tqdm(range(1, n_epochs + 1))
        self._print("Training...")
        self._print('------------------------------------------------')
        for epoch in epoch_iter:
            self._print(f'Epoch: {epoch} / {n_epochs}')
            # train
            t_start = time.time()
            train_loss, train_targets, train_preds = self._train_epoch(dataloaders['train'])
            # train_loss, train_targets, train_preds = self._evaluate(dataloaders['train'])
            score_name, train_score = scorer(train_targets, train_preds)
            train_time = time.time() - t_start
            current_lr = self._scheduler.get_last_lr()[0]
            self._print(
                f"[Train] loss: {train_loss:.3f}",
                f"{score_name}: {train_score:.3f}",
                f"lr: {current_lr:.6f}",
                f"time: {train_time:.2f}",
                sep=", "
            )
            if self._scheduler_step_after_epoch:
                self._scheduler.step()
            # valid
            t_start = time.time()
            valid_loss, valid_targets, valid_preds = self._evaluate(dataloaders['valid'])
            score_name, valid_score = scorer(valid_targets, valid_preds)
            valid_time = time.time() - t_start
            self._print(
                f"[Valid] loss: {valid_loss:.3f}",
                f"{score_name}: {valid_score:.3f}",
                f"time: {valid_time:.2f}",
                sep=", "
            )
            # save results
            result = {
                'epoch': epoch,
                'lr': current_lr,
                'train_loss': train_loss,
                f'train_{score_name}': train_score,
                'train_time': train_time,
                'valid_loss': valid_loss,
                f'valid_{score_name}': valid_score,
                'valid_time': valid_time,
            }
            result_list.append(result)
            if self._writer is not None:
                self._writer.add_scalar('Train/train_loss', train_loss, global_step=epoch)
                self._writer.add_scalar('Train/valid_loss', valid_loss, global_step=epoch)
                self._writer.add_scalar(f'Train/train_{score_name}', train_score, global_step=epoch)
                self._writer.add_scalar(f'Train/valid_{score_name}', valid_score, global_step=epoch)
            if valid_score > self._best_score:
                self._best_score = valid_score
                if save_checkpoint and valid_score > score_th:
                    self._save_checkpoint()
            self._print('------------------------------------------------')
            if epoch == break_after_epoch:
                break
        self._writer.close()
        self._print('Finished training!')
        return result_list

    def _train_epoch(self, dataloader):
        """
        dataloader: training dataloader
        returns: train loss
        """
        self._model.train()
        total_loss = 0.0
        targets_list = []
        preds_list = []
        self._optimizer.zero_grad()
        for input_ids, targets, uids in dataloader:
            targets_list.append(targets.detach().cpu())
            input_ids = input_ids.to(self._device)
            targets = targets.to(self._device)
            logits = self._model(input_ids)
            # get loss
            if self._label_smoothing or self._weighted_loss:
                uids = uids.to(self._device)
                loss = self._criterion(logits.squeeze(-1), targets, uids)
            else:
                loss = self._criterion(logits.squeeze(-1), targets)
            preds_list.append(logits.detach().cpu())
            current_lr = self._scheduler.get_last_lr()
            self._writer.add_scalar('Train (details)/train_loss', loss.item(), self._n_iter_train)
            self._writer.add_scalars(
                'Train (details)/lr',
                {f'group[{group_id}]': lr for group_id, lr in enumerate(current_lr)},
                self._n_iter_train
            )
            total_loss += loss.item()
            loss = loss / self._num_accum_steps
            loss.backward()
            self._n_iter_train += 1
            if self._n_iter_train % self._num_accum_steps == 0:
                if self._max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._optimizer.step()
                if not self._scheduler_step_after_epoch:
                    self._scheduler.step()
                self._optimizer.zero_grad()
        dl_targets = torch.cat(targets_list).numpy()
        dl_preds = torch.cat(preds_list).numpy()
        return total_loss / len(dataloader), dl_targets, dl_preds

    def _evaluate(self, dataloader):
        """
        dataloader: evaluation dataloader
        returns: eval loss
        """
        self._model.eval()
        total_loss = 0.0
        targets_list = []
        preds_list = []
        with torch.no_grad():
            for input_ids, targets, uids in dataloader:
                targets_list.append(targets.detach().cpu())
                input_ids = input_ids.to(self._device)
                targets = targets.to(self._device)
                logits = self._model(input_ids)
                # get loss
                loss = self._criterion(logits.squeeze(-1), targets)
                preds_list.append(logits.detach().cpu())
                total_loss += loss.item()
            dl_targets = torch.cat(targets_list).numpy()
            dl_preds = torch.cat(preds_list).numpy()
        return total_loss / len(dataloader), dl_targets, dl_preds

    def _save_checkpoint(self):
        torch.save(self._model.state_dict(), self._checkpoint_path)
