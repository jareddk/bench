from fielder import FieldClass
import torch
from typing import List


class DatasetBase(FieldClass):
    def get_batch(self):
        pass

    @property
    def data(self):
        while True:
            yield self.get_batch()


class RandomRealData(DatasetBase):
    shape: List[int]
    batch_size: int = 1
    repeat: bool = True
    position: int = 0

    def __post_init__(self):
        super().__post_init__()
        self._data = torch.rand(*self.shape)
        self._y = torch.rand(self.shape[0])

    def wrap(self, x):
        result = x[self.position : self.position + self.batch_size]
        if self.position + self.batch_size < self.shape[0]:
            return result
        return torch.cat([result, x[: self.position + self.batch_size - self.shape[0]]])

    def get_batch(self):
        result = {"input": self.wrap(self._data), "y": self.wrap(self._y)}
        self.position = (self.position + self.batch_size) % self.shape[0]
        return result
