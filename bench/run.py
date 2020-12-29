from fielder import FieldClass
from typing import Iterable, List, Dict
from bench.triggers import TriggerBase, NeverTrigger
from bench.models import ModelBase
from bench.losses import LossBase
from bench.hooks import HookBase
from bench.datasets import DatasetBase
import torch


class RunBase(FieldClass):
    def go(self):
        pass


class SupervisedTrainRun(RunBase):
    dataset: DatasetBase
    model: ModelBase
    hooks: List[HookBase]
    loss: LossBase
    optimizer: torch.optim.Optimizer
    optimizer_kwargs: Dict = None
    end_trigger: TriggerBase = NeverTrigger()

    def __post_init__(self):
        super().__post_init__()
        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        self.optimizer = self.optimizer(
            self.model.parameters(), **self.optimizer_kwargs
        )
        for hook in self.hooks:
            hook.setup()
        self.results = []

    def train_one_step(self, data_batch: Dict, model_outputs: Dict, step_results: Dict):
        train_loss = self.loss.loss_fn(data_batch, model_outputs)
        step_results["train_loss"] = train_loss

        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()
        return step_results

    def go(self):
        for step, batch in enumerate(self.dataset.data):
            step_results = {"step": step}
            model_outputs = self.model.forward(batch)
            step_results = self.train_one_step(batch, model_outputs, step_results)

            for hook in self.hooks:
                step_results.update(
                    hook.post_step(
                        run_state=step_results,
                        data_batch=batch,
                        model_outputs=model_outputs,
                    )
                )
            self.results.append(step_results)

            if self.end_trigger.trigger(step_results):
                break
