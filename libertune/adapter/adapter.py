
import torch
import torch.nn as nn
from libertune.model.mistral.blocks import  (

    MistralRMSNorm,
    MistralDecoderLayer,
    DynamicCache,
    Cache
)
from typing import *





class Adapter(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=9999,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        sliding_window=4096,
        attention_dropout=0.0, 
        attn_implementation= "sdpa",
        tokenizer = 'mistralai/Mistral-7B-v0.1',
        path = None,
        output_attentions = False,
        output_hidden_states = False,
        use_return_dict = True,
        device = "cpu"
        ):
        super().__init__()
        self.device = device
        self.to(device)
        config = self.set_config(locals())
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.attn_implementation = config.attn_implementation
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.load_weights(config.path)
        self.set_tokenizer(config.tokenizer)

    def set_config(self, config):
        from munch import Munch

        config = dict(config)
        config.pop('self')
        self.config = c.dict2munch(config)
        return self.config
    def load_weights(self, path=None):
        if path == None:
            return self.init_weights()
        self.load_state_dict(torch.load(path))
        return {"status": "success"}

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        client_input: str = 'hey whats up' ,
        server_input: str = 'what is up',
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        # [batch_size, seq_len, emb_dim]
        token_embeddings = {
            'client': self.embed_tokens(client_input),
            'server': self.embed_tokens(server_input)
        }

        # embedd the tokens

        for key in token_embeddings:
            # [batch_size, seq_len, hidden_size]
            token_embeddings[key] = self.encode(token_embeddings[key])

        # [batch_size, seq_len, hidden_size]
        merged_embeddings = self.attention(token_embeddings['client'],
                                            token_embeddings['server'])

        # [batch_size, seq_len, vocab_size]
        logits = self.emb2logits(merged_embeddings)

        return logits
    

    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return {"status": "success"}
    
    def set_tokenizer(self, tokenizer='mistralai/Mistral-7B-v0.1'):
        from transformers import AutoTokenizer
        tokenizer =  AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        return {"status": "success"}

    def tokenize(self, 
                text: str = 'Whadup',
                padding=True, 
                truncation=True, 
                max_length=64,
                return_tensors='pt',
                add_special_tokens=False,
                device:str = None, 
                **kwargs) -> torch.Tensor:
        """ Returns tokenized text as torch tensor. """
        
        sample = self.tokenizer(text, padding=padding, 
                                    truncation=truncation, 
                                    max_length=max_length, 
                                    return_tensors=return_tensors,
                                    add_special_tokens=add_special_tokens, 
                                    **kwargs)  # assume tokenizer.padding_side = 'left'

        device = device if device != None else self.device
        
        sample = dict(
            input_ids= sample['input_ids'].to(device),
            attention_mask= sample['attention_mask'].to(device)
        )
        
        return sample


