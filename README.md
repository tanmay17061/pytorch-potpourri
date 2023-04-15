# pytorch-potpourri
an *assortment* of interesting deep learning and machine learning algorithms, all implemented for PyTorch and related frameworks.  

[potpourri (*noun*): a *mixture* or *medley* of things.]  

---
# Data maps
An implementation of the dataset filtering technique by [this publication](https://arxiv.org/abs/2009.10795). This is a PyTorch Lightning callback that can be plugged into an existing `pytorch_lightning.LightningModule` with minimal code changes.  

An example usage:

```diff
import pytorch_lightning as pl

def YourExistingLitClassifier(pl.LightningModule):
    def __init__(self, ...):
    	...
    ...
    def training_step(self, batch, batch_idx):
    	#existing training logic
    	
    	loss = #existing loss calculation on batch
+    	probs = #calculate probabilities on batch for each output label, to result in a 2D tensor of shape (batch_size,label_count)
-    	return loss
+     	return {'loss': loss, 'probs': probs}

	...
```

```diff
from potpourri.callbacks.lightning import DataMapsCallback
+datamaps_cb = DataMapsCallback()

...

trainer = pl.Trainer(
	...
-	callbacks=[...],
+	callbacks=[...,datamaps_cb],
	...
    )

trainer.fit(...)

+confidence,variability,correctness = datamaps_cb.get_coordinates()
```

When using this callback, **make sure that**:

1. sampling by [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is sequential (for example, by using a [`SequentialSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler)).
2. the model fits satisfiably on the dataset. The algorithm can generate uninteresting results otherwise.
3. the model is fit for at least 4 epochs for a better calculation of the statistics

An example datamaps generated on a dummy binary classification [dataset](https://github.com/ZeerakW/hatespeech) (using the [`roberta-base`](https://arxiv.org/abs/1907.11692) model):

![An example datamaps generated](https://user-images.githubusercontent.com/32801726/117187189-4883ef00-adf9-11eb-95e4-cb28750b2eb3.png)

Checkout [this](https://github.com/eliorc/tavolo) repository if you're looking for a callback in TensorFlow.

---
# Batch Accumulated Metrics
(This callback is explained in [this](https://medium.com/@tanmay17061/a7077ef8e55d) blog post. Feel free to check it out!)
Summary: PyTorch does batch-wise aggregation of metrics. This behaviour is less ideal for some metrics (eg- AUC-ROC, macro-F1, etc).
This is a PyTorch Lightning callback that can be plugged into an existing `pytorch_lightning.LightningModule` with minimal code changes.
Once done, you can track the correct metrics at the end of each train/validation epoch.

An example usage can be found in the [BatchAccumulatedMetricsCallback class implementation](src/potpourri/callbacks/lightning.py).
