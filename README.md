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

Points to take care of:  

1. Make sure the sampling by [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) is sequential (for example, by using a [`SequentialSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.SequentialSampler)).

An example datamaps generated on [a dummy dataset](https://github.com/ZeerakW/hatespeech):

![An example datamaps generated](https://user-images.githubusercontent.com/32801726/117187189-4883ef00-adf9-11eb-95e4-cb28750b2eb3.png)

Checkout [this](https://github.com/eliorc/tavolo) repository if you're looking for a callback in TensorFlow.