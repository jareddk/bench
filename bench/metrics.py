from typing import Dict
from fielder import FieldClass
from bench.input_output import BaseInputOutput
from typing import Dict
import torch


class MetricBase(FieldClass):
    def compute(
        self,
        data_batch: Dict,
        model_outputs: Dict,
    ) -> torch.Tensor:
        raise NotImplementedError

    def aggregate(self, many_values: torch.Tensor):
        return torch.mean(many_values)


class MaxError(MetricBase):
    key: str = "y"

    def compute(
        self,
        data_batch: Dict,
        model_outputs: Dict,
    ) -> torch.Tensor:
        return torch.max(torch.abs(data_batch[self.key] - model_outputs[self.key]))
