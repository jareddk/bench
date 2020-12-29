from fielder import FieldClass
from typing import Dict, Any


class TriggerBase(FieldClass):
    def trigger(self, run_state: Dict) -> bool:
        pass


class AlwaysTrigger(TriggerBase):
    def trigger(self, run_state: Dict) -> bool:
        return True


class NeverTrigger(TriggerBase):
    def trigger(self, run_state: Dict) -> bool:
        return False


class EndTrigger(TriggerBase):
    key: str = "step"
    final: int = 10_000

    def trigger(self, run_state: Dict) -> bool:
        if run_state[self.key] > self.final:
            return True
        return False


class PeriodicTrigger(TriggerBase):
    key: str = "step"
    period: int = 20

    def trigger(self, run_state: Dict) -> bool:
        if run_state[self.key] % self.period == 0:
            return True
        return False


class SchedulerBase(FieldClass):
    def current_value(self, run_state: Dict) -> Any:
        pass


# TODO - LR schedules etc
