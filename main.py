from libertune.model.mistral import *
from libertune.adapter import *
# from libertune.model import MistralModel
# from libertune.adapter import SwappapleAdapter

model = MistralModel(num_hidden_layers=1, intermediate_size=10)

adapter = SwappapleAdapter()

input = "whadup fam "
server_response = model.forward(input) # string
print(server_response)

server_embedding = model.get_embedding(input) # numpy array



print(model_embedding.shape)
