"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # Building a token-to-id dict with specials first, then corpus tokens in first-seen order.
    vocab: dict[str, int] = {}
    
    # Add special tokens
    for token in specials:
        if token not in vocab:
            vocab[token] = len(vocab)
            
    # Add words 
    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)
                
    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # Building the inverse id-to-token dictionary from token_to_id
    ids={}
    for token, id in token_to_id.items():
        ids[id]=token
    return ids

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # Converting whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    list_of_ids=[]

    for token in sentence.split():
        if token not in token_to_id:
            list_of_ids.append(token_to_id[unk_token])
        else:
            list_of_ids.append(token_to_id[token])

    return list_of_ids

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # Mapping each id in ids to its token string via id_to_token and return the list
    tokens_list=[]
    
    for id in ids:
        tokens_list.append(id_to_token[id])

    return tokens_list

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # Return a list of length exactly max_len, padding with pad_id or truncating.
    list_max_len=[]
    n=len(ids)
    i=0

    while len(list_max_len)<max_len:
        if i<n:
            list_max_len.append(ids[i])
        else:
            list_max_len.append(pad_id)
        i+=1

    return list_max_len

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # Stacking padded id sequences into a (B, L) torch.long tensor
    if isinstance(padded_sequences, torch.Tensor):
        return padded_sequences.long()

    if padded_sequences and isinstance(padded_sequences[0], torch.Tensor):
        return torch.stack(padded_sequences).long()

    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # Rescaling embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch
import math

def compute_positional_div_term(d_model):
    # Return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    even_indices = torch.arange(0, d_model, 2).float()
    
    div_term = torch.exp(even_indices * (-math.log(10000.0) / d_model))
    
    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # Build a column vector of position indices from 0 to max_len-1
    position_indices = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

    return position_indices

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # Writing sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # Fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # Building the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len, d_model)
    
    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)

    fill_even_indices_with_sin(pe, position, div_term)
    fill_odd_indices_with_cos(pe, position, div_term)
    
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # Adding the first L rows of positional_encoding to embedded_batch and return the sum.
    seq_len = embedded_batch.size(1)

    return embedded_batch + positional_encoding[:seq_len, :]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # Building a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # Combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # Matmul query with the transpose of key over the last two axes
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # Divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # Replacing blocked positions of scores with negative infinity
    return scores.masked_fill(~mask, float("-inf"))

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # Softmax over the last axis, zeroing rows that are entirely -inf
    weights = torch.softmax(masked_scores, dim=-1)
    return torch.nan_to_num(weights, nan=0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # Combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # Chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    scores = compute_raw_attention_scores(query, key)
    d_k = key.size(-1)
    scaled_scores = scale_attention_scores(scores, d_k)

    if mask is not None:
        scaled_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)

    attention_weights = softmax_attention_weights(scaled_scores)

    context = apply_attention_weights_to_values(attention_weights, value)

    return (context, attention_weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # Reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    b, l, d_model = tensor.shape
    return tensor.view(b, l, num_heads, d_model // num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # Rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # Merge the head axis back into the feature axis to reconstruct d_model
    b, _, l, _ = multi_head_tensor.shape
    return multi_head_tensor.transpose(1, 2).contiguous().view(b, l, -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # Return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    out = x @ weight.t()
    if bias is not None:
        out = out + bias
    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # Project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # Spliting each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    q_head = transpose_heads_before_sequence(split_last_dim_into_heads(q, num_heads))
    k_head = transpose_heads_before_sequence(split_last_dim_into_heads(k, num_heads))
    v_head = transpose_heads_before_sequence(split_last_dim_into_heads(v, num_heads))
    return (q_head, k_head, v_head)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # Running scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask=mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # Merge the head axis back into d_model and apply the output linear projection.
    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # Project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    
    # 1. Linear projections (without biais)
    q = apply_linear_projection(query, w_q, None)
    k = apply_linear_projection(key, w_k, None)
    v = apply_linear_projection(value, w_v, None)

    # 2. Reshape into per-head views : (B, num_heads, L, d_k)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)

    # 3. Run scaled dot-product attention across all heads
    context, _ = multi_head_scaled_dot_product_attention(
        q_h, k_h, v_h, mask=mask
    )

    # 4. Merge the heads back together and apply the output projection
    output = merge_heads_and_project_output(context, w_o, None)

    return output

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # Project x by w1, add b1, then apply a ReLU activation.
    linear_out = torch.matmul(x, w1) + b1
    return torch.nn.functional.relu(linear_out)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # Project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return torch.matmul(hidden, w2) + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # Compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # Return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, keepdim=True, correction=0)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # Standardize x along the last axis then apply gamma and beta affine transform
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    return gamma * x_normalized + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # Combine the residual with the sublayer output and layer-normalize the result.
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps=eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # Multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return (x * keep_mask.to(dtype=x.dtype)) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # Run multi-head self-attention on x and wrap with residual add-and-norm.
    attention_out = assemble_multi_head_attention_forward(
        x, x, x, w_q, w_k, w_v, w_o, num_heads, mask=src_mask
    )

    return apply_residual_add_and_norm(x, attention_out, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # Run the position-wise FFN on x and wrap it with residual add-and-norm.
    ffn_out = position_wise_feed_forward_network( x, w1, b1, w2, b2 )

    return apply_residual_add_and_norm( x, ffn_out, gamma, beta )

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # Chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    attn_out = encoder_layer_self_attention_sublayer(
        x=x,
        w_q=layer_params["w_q"],
        w_k=layer_params["w_k"],
        w_v=layer_params["w_v"],
        w_o=layer_params["w_o"],
        gamma=layer_params["attn_gamma"],
        beta=layer_params["attn_beta"],
        num_heads=num_heads,
        src_mask=src_mask,
    )

    out = encoder_layer_feed_forward_sublayer(
        x=attn_out,
        w1=layer_params["w1"],
        b1=layer_params["b1"],
        w2=layer_params["w2"],
        b2=layer_params["b2"],
        gamma=layer_params["ffn_gamma"],
        beta=layer_params["ffn_beta"],
    )

    return out

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # Sequentially apply each encoder layer to the running hidden state and return the final tensor.
    hidden_state = x

    for layer_params in encoder_layer_params_list:
        hidden_state = assemble_encoder_layer(
            hidden_state,
            layer_params,
            num_heads,
            src_mask=src_mask,
        )

    return hidden_state

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 68 - compute_adam_parameter_update (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

