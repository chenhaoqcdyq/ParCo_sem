import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                dual_head_flag=False,
                semantic_len=50,
                num_parts=6):
        super().__init__()
        
        if dual_head_flag:
            self.trans_base = CrossCondTransMultipartBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len, num_parts)
            self.trans_head = CrossCondTransMultipartHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len, num_parts)
        else:
            self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_parts=num_parts)
            self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, num_parts=num_parts)
        self.block_size = block_size
        self.num_vq = num_vq
        self.dual_head_flag = dual_head_flag
        self.semantic_len = semantic_len
        self.num_parts = num_parts
        # Define token ID constants
        # Assuming self.num_vq is the STOP_TOKEN_ID for the combined (semantic+separator+reconstruction) vocabulary.
        # Example: self.num_vq = 1025 means tokens 0-1024 are valid, 1025 is STOP.
        # If semantic and reconstruction codebooks are each 512:
        # Semantic tokens: 0-511
        # Separator token: 512
        # Reconstruction tokens: 513-1024
        # This implies _assumed_codebook_part_size = 512.
        # We derive _assumed_codebook_part_size from self.num_vq to be robust.
        # (self.num_vq - 1 (separator) - 1 (stop_token_id_is_size_not_index)) / 2
        # If self.num_vq = 1025 (stop token ID), then max active token index is 1024.
        # Number of active tokens = self.num_vq.
        # If vocabulary is [sem (N), sep (1), recon (N)], total active tokens = 2N+1. Stop token is 2N+1.
        # So, self.num_vq = 2N+1. N = (self.num_vq - 1) / 2.
        _assumed_codebook_part_size = (self.num_vq - 1) // 2 # e.g. (1025-1)//2 = 512

        self.SEMANTIC_TOKEN_END_IDX = _assumed_codebook_part_size - 1
        self.SEPARATOR_TOKEN_IDX = _assumed_codebook_part_size
        self.RECONSTRUCTION_TOKEN_START_IDX = _assumed_codebook_part_size + 1
        # RECONSTRUCTION_TOKEN_END_IDX is self.num_vq - 1 (the max token ID before stop)
        # STOP_TOKEN_ID is self.num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, semantic_valid_lengths=None):
        # idxs: (B, T_sequence), contains semantic (potentially padded) and reconstruction tokens
        # clip_feature: (B, C_clip)
        # semantic_valid_lengths: (B,), actual length of semantic tokens in idxs for each batch item.
        # self.semantic_len: max length of the semantic part within idxs.
        if isinstance(idxs, list):
            idxs = torch.stack(idxs)
            idxs = idxs.permute(1, 0, 2)
        B, part, T_sequence = idxs.shape
        device = idxs.device

        # 1. Create key_padding_mask for the idxs part based on semantic_valid_lengths
        # This mask is True for padded semantic tokens, False otherwise (valid semantic, reconstruction).
        key_padding_mask_for_idxs = torch.zeros_like(idxs[:,0,:], dtype=torch.bool) # Default to False (not masked)

        if semantic_valid_lengths is not None:
            for i in range(B):
                valid_sem_len = semantic_valid_lengths[i].item()
                # Determine the end of the semantic segment within idxs.
                # This is typically self.semantic_len if idxs is long enough to contain it.
                # Or it could be T_sequence if idxs is shorter than self.semantic_len.
                # The padding occurs *within* the first self.semantic_len tokens of idxs.
                actual_semantic_segment_end = self.semantic_len
                
                if valid_sem_len < actual_semantic_segment_end:
                    # Mask padded semantic tokens: from valid_sem_len to actual_semantic_segment_end
                    key_padding_mask_for_idxs[i, valid_sem_len:actual_semantic_segment_end] = True
        
        # 2. Prepare the final attention mask for the sequence seen by attention layers in trans_head.
        # This sequence is [condition_embedding, token_embeddings_from_idxs].
        final_attention_mask_for_trans_head_input = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device), # False for condition embedding
            key_padding_mask_for_idxs
        ], dim=1)

        # Pass the appropriate masks to trans_base and trans_head
        feat = self.trans_base(idxs, clip_feature, key_padding_mask=final_attention_mask_for_trans_head_input)
        logits = self.trans_head(feat, key_padding_mask=final_attention_mask_for_trans_head_input)
        
        return logits

    def sample_dual_head(self, clip_feature, if_categorial=False):
        B = clip_feature.shape[0]
        device = clip_feature.device
        # Max length of generated sequence for any part (semantic + reconstruction for that part pathway)
        # self.block_size is the max sequence length for the attention mechanism, including condition token.
        # So, max generated tokens = self.block_size - 1.
        max_gen_len = self.block_size - 1
        
        if max_gen_len <= 0:
            return torch.empty(B, self.num_parts, 0, dtype=torch.long, device=device)

        # xs will store [semantic_tokens_repeated_for_all_parts | part_specific_reconstruction_tokens]
        # Shape: (B, num_parts, current_generated_length)
        xs = torch.empty(B, self.num_parts, 0, dtype=torch.long, device=device)

        # Flags and trackers
        # True if semantic part (first self.semantic_len slots) for batch item b is considered complete
        # (either hit self.num_vq or filled self.semantic_len slots)
        semantic_generation_complete_flags = torch.zeros(B, dtype=torch.bool, device=device)
        # Stores the actual length of semantic content before padding or stop token for each batch item
        actual_semantic_lengths = torch.full((B,), self.semantic_len, dtype=torch.long, device=device)
        
        # True if reconstruction for part p of batch item b has stopped
        reconstruction_part_stopped_flags = torch.zeros(B, self.num_parts, dtype=torch.bool, device=device)
        
        # Token definitions
        stop_token = self.num_vq
        padding_token = self.num_vq + 1 # Used for padding after a sequence/part stops

        for k_idx in range(max_gen_len): # k_idx is the index of the token we are currently generating for the sequence
            # Overall stop condition
            if torch.all(semantic_generation_complete_flags) and torch.all(reconstruction_part_stopped_flags):
                break

            # Get logits for the current step k_idx
            # self.forward expects input of shape (B, num_parts, k_idx)
            # It returns a list of logits, where list_logits[p] is (B, k_idx, V)
            # We use list_logits[p][:, -1, :] to predict token at index k_idx for part p's pathway
            list_logits_all_parts = self.forward(xs, clip_feature, semantic_valid_lengths=actual_semantic_lengths)
            
            # Prepare a column of next tokens to be appended to xs
            next_tokens_column = torch.full((B, self.num_parts), padding_token, dtype=torch.long, device=device)

            # Phase 1: Semantic Token Generation (for k_idx < self.semantic_len)
            if k_idx < self.semantic_len:
                # Semantic predictions are taken from the 0-th part's logits pathway,
                # as CrossCondTransMultipartHead ensures semantic logits are consistent there.
                logits_for_semantic_token_at_k = list_logits_all_parts[0][:, -1, :] # (B, V)
                probs_semantic = F.softmax(logits_for_semantic_token_at_k, dim=-1)
                
                sampled_semantic_tokens_for_slot_k = torch.zeros(B, dtype=torch.long, device=device)
                if if_categorial:
                    dist_sem = Categorical(probs_semantic)
                    sampled_semantic_tokens_for_slot_k = dist_sem.sample()
                else:
                    _, topk_sem = torch.topk(probs_semantic, k=1, dim=-1)
                    sampled_semantic_tokens_for_slot_k = topk_sem.squeeze(-1)

                for b_idx in range(B):
                    if not semantic_generation_complete_flags[b_idx]:
                        token_val = sampled_semantic_tokens_for_slot_k[b_idx].item()
                        # Store this semantic token in ALL parts for this batch item at current semantic slot k_idx
                        next_tokens_column[b_idx, :] = token_val 
                        actual_semantic_lengths[b_idx] = k_idx + 1 # Update true length of semantic content

                        if token_val == stop_token or (k_idx == self.semantic_len - 1):
                            semantic_generation_complete_flags[b_idx] = True
                            # If stopped early, actual_semantic_lengths already updated.
                            # If stopped exactly at self.semantic_len-1, flag is also set.
                    else:
                        # Semantic generation already completed for this batch item (e.g. early stop)
                        # Fill all parts with padding token for this semantic slot k_idx
                        next_tokens_column[b_idx, :] = padding_token
            
            # Phase 2: Reconstruction Token Generation (for k_idx >= self.semantic_len)
            # This executes if k_idx is a reconstruction slot.
            # If k_idx < self.semantic_len, this block is skipped, and next_tokens_column is already filled by semantic logic.
            if k_idx >= self.semantic_len:
                for p_idx in range(self.num_parts): # Iterate over each reconstruction part pathway
                    # Logits for part p_idx at reconstruction index (k_idx - self.semantic_len)
                    logits_for_recon_token_part_p = list_logits_all_parts[p_idx][:, -1, :] # (B, V)
                    probs_recon_p = F.softmax(logits_for_recon_token_part_p, dim=-1)

                    sampled_recon_tokens_for_part_p = torch.zeros(B, dtype=torch.long, device=device)
                    if if_categorial:
                        dist_rec_p = Categorical(probs_recon_p)
                        sampled_recon_tokens_for_part_p = dist_rec_p.sample()
                    else:
                        _, topk_rec_p = torch.topk(probs_recon_p, k=1, dim=-1)
                        sampled_recon_tokens_for_part_p = topk_rec_p.squeeze(-1)

                    for b_idx in range(B):
                        # Ensure semantic phase is complete for this batch item before starting its reconstruction
                        if semantic_generation_complete_flags[b_idx]:
                            if not reconstruction_part_stopped_flags[b_idx, p_idx]:
                                token_val = sampled_recon_tokens_for_part_p[b_idx].item()
                                next_tokens_column[b_idx, p_idx] = token_val
                                if token_val == stop_token:
                                    reconstruction_part_stopped_flags[b_idx, p_idx] = True
                            # else: reconstruction for this part already stopped, next_tokens_column[b_idx, p_idx] remains padding_token
                        else:
                            # Semantic phase not yet complete for this batch item,
                            # so its reconstruction slots should also be padding.
                            # (This path should ideally not be hit if logic is correct, as semantic_generation_complete_flags
                            #  should be true for all b_idx once k_idx >= self.semantic_len, unless semantic_len is 0)
                            next_tokens_column[b_idx, p_idx] = padding_token 
                
            # Append the generated column of tokens to xs
            xs = torch.cat((xs, next_tokens_column.unsqueeze(2)), dim=2)
        
        return xs
    
    def sample_original_backup(self, clip_feature, if_categorial=False):
        """
        Original sampling logic, used when semantic_flag=False.
        Samples from the full vocabulary space. self.num_vq is the stop token.
        """
        xs = None
        for k_loop_idx in range(self.block_size):
            current_input_for_transformer = xs if xs is not None else [] 
            
            logits = self.forward(current_input_for_transformer, clip_feature)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)

            idx_sampled_token = None
            if if_categorial:
                dist = Categorical(probs)
                idx_val = dist.sample() 
                if idx_val == self.num_vq: 
                    break 
                idx_sampled_token = idx_val.unsqueeze(-1) 
            else:
                _, idx_val_topk = torch.topk(probs, k=1, dim=-1) 
                if idx_val_topk[0,0] == self.num_vq: 
                    break 
                idx_sampled_token = idx_val_topk
            
            if xs is None:
                xs = idx_sampled_token
            else:
                xs = torch.cat((xs, idx_sampled_token), dim=1)
            
        if xs is None: 
            return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
        return xs

    def sample(self, clip_feature, if_categorial=False):
        if self.dual_head_flag:
            return self.sample_dual_head(clip_feature, if_categorial)
        else:
            return self.sample_original_backup(clip_feature, if_categorial)

    def sample_batch(self, clip_feature, if_categorial=False):
        B = clip_feature.shape[0]
        device = clip_feature.device
        
        xs = torch.empty(B, self.num_parts, 0, dtype=torch.long, device=device)
        finished_sequences = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Max length of generated sequence xs is self.block_size - 1
        # because total input to attention is xs + condition_token, 
        # which must be <= self.block_size (attention/position embedding capacity)
        max_gen_len = self.block_size - 1 
        
        if max_gen_len <= 0: # Should not happen with typical block_size > 1
             return xs 

        # Assuming num_vq is the stop token (e.g. 1024).
        # num_vq + 1 is used as padding token for completed sequences.
        # Embedding tables are sized for num_vq + 2.
        padding_token_id = self.num_vq + 1 

        if self.dual_head_flag:
            # Initialize flags and trackers for dual_head multi-part generation
            semantic_generation_complete_flags = torch.zeros(B, dtype=torch.bool, device=device)
            actual_semantic_lengths = torch.full((B,), self.semantic_len, dtype=torch.long, device=device)
            # For dual_head_flag, num_parts corresponds to the number of reconstruction pathways/heads
            # from CrossCondTransMultipartHead, which aligns with self.num_parts.
            reconstruction_part_stopped_flags = torch.zeros(B, self.num_parts, dtype=torch.bool, device=device)
            
            stop_token = self.num_vq
            # padding_token_id is already defined outside the if/else

            for k_idx in range(max_gen_len): # k_idx is the index of the token we are currently generating
                # Overall stop condition for the batch
                if torch.all(semantic_generation_complete_flags) and torch.all(reconstruction_part_stopped_flags):
                    break
                
                # Get logits for the current step k_idx
                # self.forward for dual_head=True returns a list of logits from CrossCondTransMultipartHead
                # Each element list_logits_all_parts[p] is (B, current_xs_len + 1, V)
                list_logits_all_parts = self.forward(xs, clip_feature, semantic_valid_lengths=actual_semantic_lengths)
                
                # Prepare a column of next tokens to be appended to xs
                next_tokens_column = torch.full((B, self.num_parts), padding_token_id, dtype=torch.long, device=device)

                # Phase 1: Semantic Token Generation (for k_idx < self.semantic_len)
                if k_idx < self.semantic_len:
                    # Semantic predictions are taken from the 0-th part's logits pathway.
                    # CrossCondTransMultipartHead's output list_logits_all_parts[0] gives this.
                    logits_for_semantic_token_at_k = list_logits_all_parts[0][:, :, -1, :] # (B, V)
                    probs_semantic = F.softmax(logits_for_semantic_token_at_k, dim=-1)
                    
                    sampled_semantic_tokens_for_slot_k = torch.zeros(B, dtype=torch.long, device=device)
                    if if_categorial:
                        dist_sem = Categorical(probs_semantic)
                        sampled_semantic_tokens_for_slot_k = dist_sem.sample()
                    else:
                        _, topk_sem = torch.topk(probs_semantic, k=1, dim=-1)
                        sampled_semantic_tokens_for_slot_k = topk_sem.squeeze(-1)

                    for b_idx in range(B):
                        if not semantic_generation_complete_flags[b_idx]:
                            token_val = sampled_semantic_tokens_for_slot_k[b_idx]
                            # Store this semantic token in ALL parts for this batch item at current semantic slot k_idx
                            next_tokens_column[b_idx, :] = token_val 
                            actual_semantic_lengths[b_idx] = k_idx + 1 # Update true length of semantic content

                            if stop_token in token_val or (k_idx == self.semantic_len - 1):
                                semantic_generation_complete_flags[b_idx] = True
                        else:
                            # Semantic generation already completed for this batch item (e.g. early stop)
                            # Fill all parts with padding token for this semantic slot k_idx
                            next_tokens_column[b_idx, :] = padding_token_id
                
                # Phase 2: Reconstruction Token Generation (for k_idx >= self.semantic_len)
                if k_idx >= self.semantic_len:
                    for p_idx in range(self.num_parts): # Iterate over each reconstruction part pathway
                        # Logits for part p_idx at reconstruction index (k_idx - self.semantic_len)
                        logits_for_recon_token_part_p = list_logits_all_parts[p_idx][:, -1, :] # (B, V)
                        probs_recon_p = F.softmax(logits_for_recon_token_part_p, dim=-1)

                        sampled_recon_tokens_for_part_p = torch.zeros(B, dtype=torch.long, device=device)
                        if if_categorial:
                            dist_rec_p = Categorical(probs_recon_p)
                            sampled_recon_tokens_for_part_p = dist_rec_p.sample()
                        else:
                            _, topk_rec_p = torch.topk(probs_recon_p, k=1, dim=-1)
                            sampled_recon_tokens_for_part_p = topk_rec_p.squeeze(-1)

                        for b_idx in range(B):
                            # Ensure semantic phase is complete for this batch item before starting its reconstruction
                            if semantic_generation_complete_flags[b_idx]:
                                if not reconstruction_part_stopped_flags[b_idx, p_idx]:
                                    token_val = sampled_recon_tokens_for_part_p[b_idx].item()
                                    next_tokens_column[b_idx, p_idx] = token_val
                                    if token_val == stop_token:
                                        reconstruction_part_stopped_flags[b_idx, p_idx] = True
                                # else: reconstruction for this part already stopped or semantic not done,
                                # next_tokens_column[b_idx, p_idx] remains padding_token_id by default.
                            # else: (semantic_generation_complete_flags[b_idx] is False)
                                # This part pathway remains padding if semantic isn't done. Covered by default fill.
                
                # Append the generated column of tokens to xs
                # xs shape: (B, num_parts, current_k_idx)
                # next_tokens_column shape: (B, num_parts)
                # Unsqueeze to (B, num_parts, 1) for concatenation along dim 2.
                xs = torch.cat((xs, next_tokens_column.unsqueeze(2)), dim=2)

        else: # Not dual_head_flag (logic similar to sample_original_backup but with robust batch completion)
            for _ in range(max_gen_len):
                if torch.all(finished_sequences):
                    break
                
                logits_output = self.forward(xs, clip_feature, semantic_valid_lengths=None)

                if isinstance(logits_output, list):
                    # This branch is taken if self.trans_head.forward returns a list (e.g., modified CrossCondTransHead)
                    if not logits_output:
                        raise ValueError("Received empty list of logits from self.forward when expecting multiple parts.")

                    last_step_logits_from_parts = []
                    for part_logits in logits_output:
                        if part_logits.ndim > 1 and part_logits.shape[1] > 0: # Check if tensor is not empty and has time dimension
                            last_step_logits_from_parts.append(part_logits[:, -1, :])
                        # else: part_logits might be empty or not have the expected structure, skip it for averaging
                    
                    if not last_step_logits_from_parts: # No valid part logits found
                        raise ValueError("Could not extract any valid last-step logits from the list of part logits.")
                    
                    # Average the logits from all parts that provided a last step
                    logits_last_step = torch.stack(last_step_logits_from_parts) # Shape: (B, V)
                    logits_last_step = logits_last_step.permute(1, 0, 2) # Shape: (B, num_parts, V)

                elif torch.is_tensor(logits_output):
                    if logits_output.ndim < 2 or logits_output.shape[1] == 0: # Check for at least 2 dims and non-empty time dimension
                         raise ValueError("Logits tensor from self.forward has an invalid shape or zero length in the time dimension.")
                    logits_last_step = logits_output[:, -1, :] # Shape: (B, V)
                else:
                    raise TypeError(f"self.forward returned an unexpected type: {type(logits_output)}")

                probs = F.softmax(logits_last_step, dim=-1)

                next_tokens_sampled_from_dist = torch.zeros(B, self.num_parts, dtype=torch.long, device=device)
                if if_categorial:
                    dist = Categorical(probs)
                    next_tokens_sampled_from_dist = dist.sample()
                else:
                    _, next_tokens_sampled_from_dist = torch.topk(probs, k=1, dim=-1)
                    next_tokens_sampled_from_dist = next_tokens_sampled_from_dist.squeeze(-1)
                
                actual_next_tokens = torch.full((B, self.num_parts), padding_token_id, dtype=torch.long, device=device)
                end_token_id = torch.full((self.num_parts, ), self.num_vq, dtype=torch.long, device=device)

                for i in range(B):
                    if finished_sequences[i]:
                        continue
                    
                    token_i = next_tokens_sampled_from_dist[i]
                    if self.num_vq in token_i: # Sequence finished
                        finished_sequences[i] = True
                        actual_next_tokens[i] = end_token_id
                    else:
                        actual_next_tokens[i] = token_i # Append the sampled token (could be num_vq)
                
                xs = torch.cat((xs, actual_next_tokens.unsqueeze(2)), dim=2)
                
        return xs

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask to ensure attention is only to the left (and current position)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply key_padding_mask to prevent attention to padded key positions
        if key_padding_mask is not None:
            # key_padding_mask has shape (B, T) or (B, T_key) where T_key == T in self-attention
            # It should be True for positions that are padded and should be masked.
            # We need to expand it to (B, 1, 1, T) to be broadcastable with att (B, nh, T, T)
            expanded_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
            att = att.masked_fill(expanded_key_padding_mask, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_parts=6):
        super().__init__()
        self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 3, embed_dim) for _ in range(num_parts)])
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        self.num_parts = num_parts
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, key_padding_mask=None):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, part, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = torch.stack([self.tok_emb[i](idx[:, i, :]) for i in range(self.num_parts)], dim=0).sum(dim=0)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x

class CrossCondTransDualBase(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50):
        super().__init__()
        self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 2, embed_dim) for _ in range(2)])
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.semantic_len = semantic_len
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, key_padding_mask=None):
        # key_padding_mask (here named key_padding_mask_for_idxs for clarity within this function)
        # is for `idx` (shape B, T_idx). True for padded positions in idx, especially in the semantic part.
        key_padding_mask_for_idxs = key_padding_mask 

        condition_embedding = self.cond_emb(clip_feature).unsqueeze(1) # (B, 1, C)
        mask_for_blocks = None

        if len(idx) == 0:
            token_embeddings = condition_embedding
            # Mask for blocks is just for the condition embedding (i.e., False, do not mask)
            mask_for_blocks = torch.zeros(condition_embedding.shape[0], 1, dtype=torch.bool, device=condition_embedding.device)
        else:
            b, t_idx = idx.size()
            assert t_idx <= self.block_size, f"Cannot forward, idx sequence length {t_idx} exceeds model block size {self.block_size}."

            # Prepare token embeddings from idx based on whether it's semantic, reconstruction, or both
            if t_idx <= self.semantic_len : # Only semantic tokens (or shorter than self.semantic_len)
                token_embeddings_unconditioned = self.tok_emb[0](idx)
            else: # Both semantic and reconstruction tokens
                token_sem_embeddings = self.tok_emb[0](idx[..., :self.semantic_len])
                token_recon_embeddings = self.tok_emb[1](idx[..., self.semantic_len:t_idx]) # Slice up to t_idx
                token_embeddings_unconditioned = torch.cat([token_sem_embeddings, token_recon_embeddings], dim=1)
            
            token_embeddings = torch.cat([condition_embedding, token_embeddings_unconditioned], dim=1)
            
            # Adjust key_padding_mask_for_idxs for the prepended condition embedding
            if key_padding_mask_for_idxs is not None:
                # Ensure key_padding_mask_for_idxs matches the length of idx used for embeddings
                key_padding_mask_for_idxs = key_padding_mask_for_idxs[:, :t_idx + 1]
                # mask_for_blocks = torch.cat([
                #     # torch.zeros(b, 1, dtype=torch.bool, device=idx.device), # False for condition
                #     key_padding_mask_for_idxs[:, :t_idx] # Use mask corresponding to actual idx length
                # ], dim=1)
            else:
                # If no original mask, then no padding for any part of the sequence passed to blocks (beyond causal)
                # Create a mask of all Falses for the blocks if none provided.
                mask_for_blocks = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], dtype=torch.bool, device=idx.device)
            
        x = self.pos_embed(token_embeddings)
        
        for block in self.blocks:
            x = block(x, key_padding_mask=mask_for_blocks) # Pass the adjusted mask
            
        return x

class CrossCondTransMultipartBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50,
                num_parts = 6): # num_parts specifies the number of reconstruction heads
        super().__init__()
        self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 3, embed_dim) for _ in range(num_parts)])
        self.sem_emb = nn.Embedding(num_vq + 3, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.semantic_len = semantic_len
        self.num_recon_heads = num_parts
        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, key_padding_mask=None):
        # key_padding_mask (here named key_padding_mask_for_idxs for clarity within this function)
        # is for `idx` (shape B, T_idx). True for padded positions in idx, especially in the semantic part.
        key_padding_mask_for_idxs = key_padding_mask 

        condition_embedding = self.cond_emb(clip_feature).unsqueeze(1) # (B, 1, C)
        mask_for_blocks = None

        if len(idx) == 0:
            token_embeddings = condition_embedding
            # Mask for blocks is just for the condition embedding (i.e., False, do not mask)
            mask_for_blocks = torch.zeros(condition_embedding.shape[0], 1, dtype=torch.bool, device=condition_embedding.device)
        else:
            b, part_num, t_idx = idx.size()
            assert t_idx <= self.block_size, f"Cannot forward, idx sequence length {t_idx} exceeds model block size {self.block_size}."

            # Prepare token embeddings from idx based on whether it's semantic, reconstruction, or both
            if t_idx <= self.semantic_len : # Only semantic tokens (or shorter than self.semantic_len)
                token_embeddings_unconditioned = self.sem_emb(idx[:,0,:])
            else: # Both semantic and reconstruction tokens
                token_sem_embeddings = self.sem_emb(idx[:,0,:self.semantic_len])
                token_recon_embeddings = [
                    self.tok_emb[i](idx[:,i,self.semantic_len:t_idx]) for i in range(self.num_recon_heads)
                ]
                token_recon_embeddings = torch.stack(token_recon_embeddings, dim=0).sum(dim=0)
                token_embeddings_unconditioned = torch.cat([token_sem_embeddings, token_recon_embeddings], dim=1)
            
            token_embeddings = torch.cat([condition_embedding, token_embeddings_unconditioned], dim=1)
            
            # Adjust key_padding_mask_for_idxs for the prepended condition embedding
            if key_padding_mask_for_idxs is not None:
                # Ensure key_padding_mask_for_idxs matches the length of idx used for embeddings
                key_padding_mask_for_idxs = key_padding_mask_for_idxs[:, :t_idx + 1]
                # mask_for_blocks = torch.cat([
                #     # torch.zeros(b, 1, dtype=torch.bool, device=idx.device), # False for condition
                #     key_padding_mask_for_idxs[:, :t_idx] # Use mask corresponding to actual idx length
                # ], dim=1)
            else:
                # If no original mask, then no padding for any part of the sequence passed to blocks (beyond causal)
                # Create a mask of all Falses for the blocks if none provided.
                mask_for_blocks = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], dtype=torch.bool, device=idx.device)
            
        x = self.pos_embed(token_embeddings)
        
        for block in self.blocks:
            x = block(x, key_padding_mask=mask_for_blocks) # Pass the adjusted mask
            
        return x

class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                num_parts=6):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_parts)])
        self.head = nn.ModuleList([nn.Linear(embed_dim, num_vq + 1, bias=False) for _ in range(num_parts)])
        self.block_size = block_size
        self.num_parts = num_parts
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, key_padding_mask=None):
        x = self.blocks(x)
        logits = []
        for i in range(self.num_parts):
            x = self.ln_f[i](x)
            logits.append(self.head[i](x))
        return logits
    
class CrossCondTransDualHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50):
        super().__init__()

        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])
        self.sem_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.recon_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.semantic_len = semantic_len
        self.num_vq = num_vq
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, key_padding_mask=None):
        # x is the output from TransBase (Dual), includes condition embedding.
        # key_padding_mask is also for x (i.e., includes a False for the condition part at the beginning).
        # We call this key_padding_mask_for_input_x for clarity within this function.
        key_padding_mask_for_input_x = key_padding_mask

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask_for_input_x) # Pass the mask to each block

        # The input x has shape (B, 1 + T_idx, C) due to condition embedding.
        # T_idx is the original length of idx tokens (semantic + reconstruction).
        # self.semantic_len is the length of semantic part in *original idxs*.
        
        # Slicing for semantic and reconstruction parts after blocks:
        # Semantic part in x (after condition): x[:, 1 : 1 + self.semantic_len, :]
        # Reconstruction part in x (after condition and semantic): x[:, 1 + self.semantic_len :, :]

        # Note: The slicing here assumes x's length is at least 1 + self.semantic_len for the semantic part,
        # and potentially longer for the reconstruction part.
        # If x is shorter, e.g., during early generation steps, these slices might be empty or out of bounds.
        # The lengths should be handled carefully based on actual sequence length of x.
        # current_seq_len_after_cond = x.shape[1]

        # Determine the end index for the semantic part in x
        # It's 1 (for condition) + length of semantic tokens in x
        # Max length of semantic tokens in x is self.semantic_len
        # Actual length of semantic tokens in x is min(self.semantic_len, current_seq_len_after_cond)
        # semantic_part_end_in_x = 1 + min(self.semantic_len, current_seq_len_after_cond)
        
        x_semantic_part = x[:, :self.semantic_len, :]
        x_recon_part = x[:, self.semantic_len:, :] # Takes the rest
        
        logits_semantic = torch.empty(x.shape[0], 0, self.sem_heads.out_features, device=x.device)
        if x_semantic_part.shape[1] > 0:
            x_semantic_out = self.ln_f[0](x_semantic_part)
            logits_semantic = self.sem_heads(x_semantic_out)
        
        logits_recon = torch.empty(x.shape[0], 0, self.recon_heads.out_features, device=x.device)
        if x_recon_part.shape[1] > 0:
            x_recon_out = self.ln_f[1](x_recon_part)
            logits_recon = self.recon_heads(x_recon_out)
        
        # The logits should correspond to the positions *after* the condition token.
        logits_result = torch.cat([logits_semantic, logits_recon], dim=1)
        return logits_result

class CrossCondTransMultipartHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50,
                num_parts = 6): # num_parts specifies the number of reconstruction heads
        super().__init__()

        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.semantic_len = semantic_len
        self.num_recon_heads = num_parts

        # Shared transformer blocks
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])

        # Semantic head
        self.ln_f_semantic = nn.LayerNorm(embed_dim)
        self.head_semantic = nn.Linear(embed_dim, num_vq + 1, bias=False)

        # Reconstruction heads
        self.ln_f_reconstruction_list = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_parts)
        ])
        self.heads_reconstruction_list = nn.ModuleList([
            nn.Linear(embed_dim, num_vq + 1, bias=False) for _ in range(num_parts)
        ])
        
        # # Calculate theoretical maximum lengths for each reconstruction segment
        # self.reconstruction_segment_lengths = []
        # # Max number of tokens (semantic + recon) that can be generated is block_size - 1 (due to condition token)
        # total_token_capacity = self.block_size - 1 
        # available_for_reconstruction = total_token_capacity - self.semantic_len
        
        # if available_for_reconstruction < 0: # Ensure it's not negative
        #     available_for_reconstruction = 0

        # if self.num_recon_heads > 0 and available_for_reconstruction > 0:
        #     base_segment_len = available_for_reconstruction // self.num_recon_heads
        #     remainder_len = available_for_reconstruction % self.num_recon_heads
        #     for i in range(self.num_recon_heads):
        #         self.reconstruction_segment_lengths.append(base_segment_len + (1 if i < remainder_len else 0))
        # else: # No space for reconstruction parts or no reconstruction heads
        #     for _ in range(self.num_recon_heads):
        #         self.reconstruction_segment_lengths.append(0)

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, key_padding_mask=None):
        # x is the output from TransBase, includes condition embedding.
        # x shape: (B, 1 + T_idx_orig, C) where T_idx_orig is original idx length.
        # key_padding_mask is for x.
        
        x_after_blocks = x
        for block_layer in self.blocks: # Renamed from self.blocks to avoid confusion in loop
            x_after_blocks = block_layer(x_after_blocks, key_padding_mask=key_padding_mask)

        B, T_full_input_len, C_dim = x_after_blocks.shape
        

        semantic_features = x_after_blocks[:, :self.semantic_len, :]
        normed_sem_features = self.ln_f_semantic(semantic_features)
        logits_semantic = self.head_semantic(normed_sem_features)

        recon_features = x_after_blocks[:, self.semantic_len:, :]
        if recon_features.shape[1] > 0:
            all_recon_logits_parts = []
            for i in range(self.num_recon_heads):
                ln_layer = self.ln_f_reconstruction_list[i]
                head_layer = self.heads_reconstruction_list[i]
                recon_features_part = recon_features
                normed_recon_features = ln_layer(recon_features_part)
                current_part_logits = head_layer(normed_recon_features)
                all_recon_logits_parts.append(current_part_logits)
            all_recon_logits_parts = torch.stack(all_recon_logits_parts).permute(1, 0, 2, 3) 
            logits_semantic = logits_semantic.unsqueeze(1).expand(-1, self.num_recon_heads, -1, -1)
            logits_result = torch.cat([logits_semantic, all_recon_logits_parts], dim=2)
            output = [logits_result[:, i, :, :] for i in range(self.num_recon_heads)]
        else:
            # all_recon_logits_parts = []
            logits_semantic = logits_semantic.unsqueeze(1).expand(-1, self.num_recon_heads, -1, -1)
            output = [logits_semantic]
        # (num_parts, B, T, num_vq + 1) -> (B, num_parts, T, num_vq + 1)
        return output

