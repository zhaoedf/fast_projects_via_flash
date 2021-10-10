
'''
    Registries are Flash internal key-value database to store a mapping between a name and a function.
'''


from flash import Task
from flash.core.registry import FlashRegistry
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

# creating a custom `Task` with its own registry
class SemanticSegmentationModelHub(Task):

    backbones = FlashRegistry("backbones")

    def __init__(
        self,
        backbone: str = "zdf/UNet",
        pretrained: bool = True,
        learning_rate = 1e-4
    ):
        super().__init__(
            learning_rate=learning_rate
            ) # official doc doesn't have this line
        self.backbone = self.backbones.get(backbone)(pretrained=pretrained)
        
        
    def step(self, batch: Any, batch_idx: int, metrics: nn.ModuleDict) -> Any:
        """The training/validation/test step.

        Override for custom behavior.
        """
        print(2)
        print(batch.keys())
        x, y = batch['input'], batch['target']
        y_hat = self.backbone(x)
        y, y_hat = self.apply_filtering(y, y_hat)
        output = {"y_hat": y_hat}
        y_hat = self.to_loss_format(output["y_hat"])
        losses = {name: l_fn(y_hat, y) for name, l_fn in self.loss_fn.items()}

        y_hat = self.to_metrics_format(output["y_hat"])

        logs = {}

        for name, metric in metrics.items():
            if isinstance(metric, torchmetrics.metric.Metric):
                metric(y_hat, y)
                logs[name] = metric  # log the metric itself if it is of type Metric
            else:
                logs[name] = metric(y_hat, y)

        if len(losses.values()) > 1:
            logs["total_loss"] = sum(losses.values())
            return logs["total_loss"], logs

        output["loss"] = self.compute_loss(losses)
        output["logs"] = self.compute_logs(logs, losses)
        output["y"] = y
        return output


from .UNet import UNet
@SemanticSegmentationModelHub.backbones(name="zdf/UNet")
def UNet512(pretrained: bool = True):
    print('1')
    model = UNet(3, 1)
    return model


print(SemanticSegmentationModelHub.available_backbones())
# model = SemanticSegmentationModelHub(backbone='zdf/UNet', learning_rate=1e-3)
