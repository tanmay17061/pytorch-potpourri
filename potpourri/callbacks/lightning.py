import numpy as np
import pytorch_lightning as pl

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
                if 'extra' not in outputs[0][0]:
                    raise ValueError()
                extra = outputs[0][0]['extra']

                if get_probs_from_outputs_key_or_callable in extra:
                    probs = extra[get_probs_from_outputs_key_or_callable]
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
