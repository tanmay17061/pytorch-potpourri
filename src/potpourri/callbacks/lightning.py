import numpy as np
import pytorch_lightning as pl
import torch

class DataMapsCallback(pl.callbacks.Callback):
    """
    A pytorch lightning callback, implementing `Data Maps` described in the publication (https://arxiv.org/abs/2009.10795).

    Args:
        get_probs_from_outputs_key_or_callable (:obj:`Union[Callable[[training_step_output], np.ndarray], str]`, `optional`):
            If:

                1. `str`: the key to be used to extract the `np.ndarray` (of shape (batch_size, num_labels),
                    where (i,j)th element contains the predicted probability of ith element belonging  to the jth label)
                    from the `outputs` (the outputs of training_step_end, as described under
                    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end).

                2. Callable[[training_step_output], np.ndarray]: a method that accepts the `outputs`
                    (the outputs of training_step_end, as described under
                    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end)
                    and returns an `np.ndarray` of shape (batch_size, num_labels),
                    where (i,j)th element contains the predicted probability of ith element belonging  to the jth label.

        The default value is "probs".

            NOTE: To use this callback, the user must modify their LightningModule's training_step() to return the
                probability `np.ndarray`.
    Example Usage:
        TBD.
    """
    def __init__(self, get_probs_from_outputs_key_or_callable="probs",get_labels_from_batch_key_or_callable="label"):

        if callable(get_probs_from_outputs_key_or_callable):
            self.get_probs_from_outputs_callable = get_probs_from_outputs_key_or_callable
        else:
            def _get_probs_from_outputs_callable(outputs):
                """
                TBD.
                """

                if get_probs_from_outputs_key_or_callable in outputs:
                    probs = outputs[get_probs_from_outputs_key_or_callable]
                    if isinstance(probs,torch.Tensor):
                        return probs.detach().cpu().numpy()
                    elif isinstance(probs,np.ndarray):
                        return probs
                    else:
                        raise ValueError()
                else:
                    raise ValueError()
            self.get_probs_from_outputs_callable = _get_probs_from_outputs_callable

        if callable(get_labels_from_batch_key_or_callable):
            self.get_labels_from_batch_callable = get_labels_from_batch_key_or_callable
        else:
            def _get_labels_from_batch_callable(batch):
                """
                TBD.
                """
                if not isinstance(batch, dict):
                    raise ValueError()
                if get_labels_from_batch_key_or_callable in batch:
                    return batch[get_labels_from_batch_key_or_callable].detach().cpu().numpy()
                else:
                    raise ValueError()
            self.get_labels_from_batch_callable = _get_labels_from_batch_callable

        self.probs_over_epochs = list()
        self.pred_correct_labels_over_epochs = list()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        batch_probs = self.get_probs_from_outputs_callable(outputs)
        batch_labels = self.get_labels_from_batch_callable(batch)

        assert batch_probs.shape[0] == batch_labels.shape[0]

        batch_pred_correct_labels = (np.argmax(batch_probs,axis=-1) == batch_labels)

        batch_label_probs = batch_probs[np.arange(batch_labels.shape[0]),batch_labels]

        self.probs_over_epochs[-1].append(batch_label_probs)
        self.pred_correct_labels_over_epochs[-1].append(batch_pred_correct_labels)

    def on_train_epoch_start(self, trainer, pl_module):

        self.probs_over_epochs.append([])
        self.pred_correct_labels_over_epochs.append([])

    def on_train_end(self, trainer, pl_module):

        probs_over_epochs = [np.concatenate(epoch_l).reshape(-1,1) for epoch_l in self.probs_over_epochs]
        pred_correct_labels_over_epochs = [np.concatenate(epoch_l).reshape(-1,1) for epoch_l in self.pred_correct_labels_over_epochs]

        probs_over_epochs = np.concatenate(probs_over_epochs,axis=1)
        pred_correct_labels_over_epochs = np.concatenate(pred_correct_labels_over_epochs,axis=1)

        self.confidence = np.mean(probs_over_epochs,axis=1)
        self.variability = np.std(probs_over_epochs,axis=1)
        self.correctness = np.mean(pred_correct_labels_over_epochs,axis=1)

    def get_coordinates(self):
        """
        TBD.
        """
        return self.confidence, self.variability, self.correctness

class BatchAccumulatedMetricsCallback(pl.callbacks.Callback):
    """
    A pytorch lightning callback to help capture true metrics across an epoch of train/validation run.

    Args:

            NOTE: To use this callback, the user must modify their LightningModule's train/validation/test steps
            to return the true_labels and pred_labels, both of type: `np.ndarray` and dimension: (batch_size,).
    
    -------------------

        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        from potpourri.callbacks.lightning import BatchAccumulatedMetricsCallback

        trainer_callbacks = [
            ModelCheckpoint(
                ...
                monitor="val_accumulated_macro_f1",
                mode="max",
                ...
            ),
            EarlyStopping(
                ...
                monitor="val_accumulated_macro_f1",
                mode="max",
                ...
            ),
            BatchAccumulatedMetricsCallback(
                metric_to_function_dict = {
                    "accumulated_macro_f1": (f1_score,{'average':'macro'}), 
                    "accumulated_weighted_macro_f1": (f1_score,{'average':'weighted'}),
                }
            ),
        ]
        trainer = Trainer(callbacks=trainer_callbacks, ...)

    -------------------
    This example tracks `val_accumulated_macro_f1` metric for model checkpointing and early stopping. NOTE: You can
    refer to (https://medium.com/@tanmay17061/a7077ef8e55d) to understand why this might be important.
    (However, The example in the blog post does not support `metric_to_function_dict` to make things more readable for a lesser technical audience.)
    """
    def __init__(self, metric_to_function_dict):
        self.metric_to_function_dict = metric_to_function_dict

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_actual_labels_npy = np.empty(shape=(0,), dtype=int)
        self.train_pred_labels_npy = np.empty(shape=(0,), dtype=int)

    def on_train_epoch_end(self, trainer, pl_module):
        for m,fn_args in self.metric_to_function_dict.items():
            fn,args = fn_args
            pl_module.log("train_" + m, fn(self.train_actual_labels_npy,self.train_pred_labels_npy,**args))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.train_actual_labels_npy = np.concatenate((self.train_actual_labels_npy,outputs["actual_labels"],))
        self.train_pred_labels_npy = np.concatenate((self.train_pred_labels_npy,outputs["pred_labels"],))

    def on_validation_start(self, trainer, pl_module):
        self.val_actual_labels_npy = np.empty(shape=(0,), dtype=int)
        self.val_pred_labels_npy = np.empty(shape=(0,), dtype=int)

    def on_validation_epoch_end(self, trainer, pl_module):
        for m,fn_args in self.metric_to_function_dict.items():
            fn,args = fn_args
            pl_module.log("val_" + m, fn(self.val_actual_labels_npy,self.val_pred_labels_npy,**args))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_actual_labels_npy = np.concatenate((self.val_actual_labels_npy,outputs["actual_labels"],))
        self.val_pred_labels_npy = np.concatenate((self.val_pred_labels_npy,outputs["pred_labels"],))