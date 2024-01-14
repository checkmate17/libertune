from libertune.model import MistralModel
from libertune.adapter import SwappapleAdapter

import commune as c
model = MistralModel(num_hidden_layers=1, intermediate_size=10)

adapter = SwappapleAdapter()

input = "whadup fam "
model_output_string = model.forward(input) # [batch, seq_len, embedding_size]



print(model_embedding.shape)
