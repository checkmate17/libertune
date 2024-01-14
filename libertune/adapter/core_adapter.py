import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(
            self,
            input_dim: int = 256,
            output_dim: int = 256,
            base_model: str = 'bert-base-uncased',
            dropout: float = 0.1):
        
        kwargs = locals()
        kwargs.pop('self')
        self.__dict__.update(kwargs)

        super().__init__()
        self.build()



    def build(self):

        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.output_dim, self.input_dim),
        )

    



