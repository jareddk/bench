import torch
from bench.input_output import BaseInputOutput, FunctionData, RealOutput
from bench.metrics import MetricBase
from typing import Dict


class LossBase(MetricBase):
    def loss_fn(
        self,
        data_batch: Dict,
        model_outputs: Dict,
    ) -> torch.Tensor:
        return self.compute(data_batch, model_outputs)

    def aggregate(self, single_values: torch.Tensor):
        return torch.mean(single_values)


class MSELoss(LossBase):
    key: str = "y"

    def compute(
        self,
        data_batch: Dict,
        model_outputs: Dict,
    ) -> torch.Tensor:
        return torch.mean((data_batch[self.key] - model_outputs[self.key]) ** 2)


#
# class LossBase(MetricBase):
#     def loss_fn(
#         self,
#         data_batch: BaseInputOutput,
#         model_outputs: BaseInputOutput,
#     ) -> torch.Tensor:
#         return self.compute(data_batch, model_outputs)
#
#     def aggregate(self, single_values: torch.Tensor):
#         return torch.mean(single_values)
#
#
# class MSELoss(LossBase):
#     def compute(
#         self,
#         data_batch: FunctionData,
#         model_outputs: RealOutput,
#     ) -> torch.Tensor:
#         return torch.mean((data_batch.y - model_outputs.y) ** 2)
