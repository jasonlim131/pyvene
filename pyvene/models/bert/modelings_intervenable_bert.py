"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the Hugging Face library.

We also want to let the intervention library know how to
configure the dimensions of intervention based on model config
defined in the Hugging Face library.
"""

import torch
from ..constants import *
from functools import partial

# Define any necessary transformation functions
def split_head_and_permute(tensor, n_heads):
    """
    Splits the tensor into multiple heads and permutes it for multi-head attention.

    Args:
        tensor (torch.Tensor): The input tensor to split and permute.
        n_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    # Assuming tensor shape: [batch_size, seq_length, hidden_size]
    new_shape = tensor.size()[:-1] + (n_heads, tensor.size(-1) // n_heads)
    tensor = tensor.view(*new_shape)
    tensor = tensor.permute(0, 2, 1, 3)
    return tensor


# Base mapping for BERT-style models
bert_type_to_module_mapping = {
    "block_input": ("layer._.%s", CONST_INPUT_HOOK),
    "block_output": ("layer._.%s.output", CONST_OUTPUT_HOOK),
    "mlp_activation": ("layer._.%s.intermediate.act_fn", CONST_OUTPUT_HOOK),
    "mlp_output": ("layer._.%s.output", CONST_OUTPUT_HOOK),
    "mlp_input": ("layer._.%s.intermediate.dense", CONST_INPUT_HOOK),
    "attention_value_output": ("layer._.%s.attention.self.value", CONST_INPUT_HOOK),
    "head_attention_value_output": (
        "layer._.%s.attention.self.value",
        CONST_INPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
    "attention_output": ("layer._.%s.attention.output", CONST_OUTPUT_HOOK),
    "attention_input": ("layer._.%s.attention.self", CONST_INPUT_HOOK),
    "query_output": ("layer._.%s.attention.self.query", CONST_OUTPUT_HOOK),
    "key_output": ("layer._.%s.attention.self.key", CONST_OUTPUT_HOOK),
    "value_output": ("layer._.%s.attention.self.value", CONST_OUTPUT_HOOK),
    "head_query_output": (
        "layer._.%s.attention.self.query",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_attention_heads")
    ),
    "head_key_output": (
        "layer._.%s.attention.self.key",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads")
    ),
    "head_value_output": (
        "layer._.%s.attention.self.value",
        CONST_OUTPUT_HOOK,
        (split_head_and_permute, "num_key_value_heads")
    ),
}

bert_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "n_kv_head": ("num_key_value_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("head_dim",),  # Assuming head_dim = hidden_size / num_attention_heads
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("head_dim",),
    "head_key_output": ("head_dim",),
    "head_value_output": ("head_dim",),
}

"""BERT-style model with LM head"""
bert_lm_type_to_module_mapping = {}
for k, v in bert_type_to_module_mapping.items():
    bert_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

bert_lm_type_to_dimension_mapping = bert_type_to_dimension_mapping

"""BERT-style model with classifier head"""
bert_classifier_type_to_module_mapping = {}
for k, v in bert_type_to_module_mapping.items():
    # Adjust the module path to include the classifier head
    # For BertForSequenceClassification, the classifier is separate from the encoder
    bert_classifier_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

bert_classifier_type_to_dimension_mapping = bert_type_to_dimension_mapping

def create_bert(
    name,
    model_type="bert",
    cache_dir=None,
    dtype=torch.bfloat16,
    config=None,
    num_labels=2
):
    if model_type == "bert":
        from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
        ModelClass = BertForSequenceClassification
        TokenizerClass = BertTokenizer
        ConfigClass = BertConfig
    
    # robert uses a BPE tokenizer
    elif model_type == "roberta":
        from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
        ModelClass = RobertaForSequenceClassification
        TokenizerClass = RobertaTokenizer
        ConfigClass = RobertaConfig
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if config is None:
        config = ConfigClass.from_pretrained(
            name,
            cache_dir=cache_dir,
            num_labels=num_labels
        )
    else:
        config.num_labels = num_labels

    tokenizer = TokenizerClass.from_pretrained(name, cache_dir=cache_dir)
    model = ModelClass.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,
    )
    
    return config, tokenizer, model
