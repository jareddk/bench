from fielder import FieldClass
from typing import Dict, Iterable
from bench.triggers import TriggerBase, AlwaysTrigger
from bench.metrics import MetricBase


class HookBase(FieldClass):
    trigger: TriggerBase = AlwaysTrigger()

    def setup(self):
        pass

    def post_step(self, data_batch: Dict, model_outputs: Dict) -> Dict:
        return run_state


class MetricHook(HookBase):
    metrics: Dict[str, MetricBase]

    def post_step(self, run_state: Dict, data_batch: Dict, model_outputs: Dict) -> Dict:
        if self.trigger.trigger(run_state):
            results = {}
            for name, metric in self.metrics.items():
                results[name] = metric.compute(data_batch, model_outputs)
            return results


class EvalHook(MetricHook):
    dataset: Iterable
    # TODO
