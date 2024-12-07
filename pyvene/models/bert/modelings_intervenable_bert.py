# intervenable_roberta.py

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

roberta_type_to_module_mapping = {
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

# The dimension mapping remains the same as it's correct
roberta_type_to_dimension_mapping = {
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_activation": ("intermediate_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size / num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size / num_attention_heads",),
    "head_key_output": ("hidden_size / num_attention_heads",),
    "head_value_output": ("hidden_size / num_attention_heads",),
}

"""RoBERTa model with LM head"""
roberta_lm_type_to_module_mapping = {}
for k, v in roberta_type_to_module_mapping.items():
    roberta_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

roberta_lm_type_to_dimension_mapping = roberta_type_to_dimension_mapping

"""RoBERTa model with classifier head"""
roberta_classifier_type_to_module_mapping = {}
for k, v in roberta_type_to_module_mapping.items():
    # Adjust the module path to include the classifier head
    # For RobertaForSequenceClassification, the classifier is separate from the encoder
    roberta_classifier_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]

roberta_classifier_type_to_dimension_mapping = roberta_type_to_dimension_mapping

def create_roberta(
    name="roberta-base",
    cache_dir=None,
    dtype=torch.bfloat16,
    config=None,
    num_labels=2  # Default number of labels for classification
):
    from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

    if config is None:
        config = RobertaConfig.from_pretrained(
            name,
            cache_dir=cache_dir,
            num_labels=num_labels
        )
    else:
        config.num_labels = num_labels

    tokenizer = RobertaTokenizer.from_pretrained(name, cache_dir=cache_dir)

    roberta = RobertaForSequenceClassification.from_pretrained(
        name,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=dtype,  # Save memory
    )
    print("loaded model")
    return config, tokenizer, roberta
