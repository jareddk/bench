import torch
from fielder import FieldClass
import yaml


class ModelBase(FieldClass, torch.nn.Module):
    """Base Model Class"""


class FCNet(ModelBase):
    d_in: int = 10
    H: int = 100
    n_hidden: int = 1
    D_out: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.input_linear = torch.nn.Linear(self.d_in, self.H)
        self.middle_linears = torch.nn.ModuleList(
            [torch.nn.Linear(self.H, self.H) for _ in range(self.n_hidden)]
        )
        self.output_linear = torch.nn.Linear(self.H, self.D_out)

    def layer_n_activation(self, x, n_layer=-1):
        x = x["input"]
        if n_layer == 0:
            return x
        x = self.input_linear(x).clamp(min=0)
        if n_layer == 1:
            return x
        for i, layer in enumerate(self.middle_linears):
            x = layer(x).clamp(min=0)
            if i + 2 == n_layer:
                return x
        return x

    def forward(self, x):
        return {"y": self.output_linear(self.layer_n_activation(x))}


class SimpleTransformer(ModelBase):
    """
    DON'T USE THIS AS A REFERENCE IMPLEMENTATION
    It's BiDir, ie it doesn't have any attention masking,
    and it hasn't been debugged carefully
    """

    d_model: int = 16
    m_mlp: int = 2
    n_head: int = 1
    n_ctx: int = 2
    n_layer: int = 2
    d_out: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.n_head * self.d_head == self.d_model

        self.layers = torch.nn.ModuleList(
            SimpleTransformerLayer(
                d_model=self.d_model,
                m_mlp=self.m_mlp,
                n_head=self.n_head,
                n_ctx=self.n_ctx,
            )
            for _ in range(self.n_layer)
        )
        self.output_linear = torch.nn.Linear(self.d_model * self.n_ctx, self.d_out)

    @property
    def d_head(self):
        return self.d_model // self.n_head

    @property
    def D_in(self):
        return self.n_ctx * self.d_model

    def torso(self, x):
        x = x.reshape(-1, self.n_ctx, self.d_model)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.torso(x).reshape(-1, self.D_in)
        return self.output_linear(x)


class SimpleTransformerLayer(ModelBase):
    d_model: int = 32
    m_mlp: int = 4
    n_head: int = 1
    n_ctx: int = 4

    def __post_init__(self):
        super().__post_init__()

        assert self.n_head * self.d_head == self.d_model

        self.mlp_linear1 = torch.nn.Linear(self.d_model, self.m_mlp * self.d_model)
        self.mlp_linear2 = torch.nn.Linear(self.m_mlp * self.d_model, self.d_model)

        self.query = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = torch.nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = torch.nn.Linear(self.d_model, self.d_model)
        self.dense = torch.nn.Linear(self.d_model, self.d_model)

    @property
    def d_head(self):
        return self.d_model // self.n_head

    def reshape_as_heads(self, x):
        new_shape = x.shape[:-1] + (self.n_head, self.d_head)
        return x.reshape(*new_shape)

    def reshape_as_d_model(self, x):
        new_shape = x.shape[:-2] + (self.d_model,)
        return x.reshape(*new_shape)

    def attn(self, x):
        q = self.reshape_as_heads(self.query(x))
        k = self.reshape_as_heads(self.key(x))
        v = self.reshape_as_heads(self.value(x))

        attn_logits = torch.einsum("bshi,bthi->bhst", q, k) / torch.sqrt(
            torch.tensor(self.d_head, dtype=torch.float)
        )
        attn_weights = torch.nn.Softmax(dim=-1)(attn_logits)
        attention_result = torch.einsum("bhst,bthi->bshi", attn_weights, v)
        result = self.reshape_as_d_model(attention_result)

        return self.dense(result)

    def mlp(self, x):
        m = self.mlp_linear1(x).clamp(0)
        return self.mlp_linear2(m)

    def forward(self, x):
        x = torch.layer_norm(x, normalized_shape=x.shape[1:])
        a = self.attn(x)
        x = torch.layer_norm(x + a, normalized_shape=x.shape[1:])
        m = self.mlp(x)
        return x + m
