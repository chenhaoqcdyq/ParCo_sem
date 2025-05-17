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
                semantic_flag=False,
                semantic_interleaved_flag=False,
                dual_head_flag=False,
                semantic_len=50):
        super().__init__()
        
        if dual_head_flag:
            self.trans_base = CrossCondTransDualBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len)
            self.trans_head = CrossCondTransDualHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len)
        else:
            self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
            self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        self.semantic_flag = semantic_flag
        self.semantic_interleaved_flag = semantic_interleaved_flag
        self.dual_head_flag = dual_head_flag
        self.semantic_len = semantic_len
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

        B, T_sequence = idxs.shape
        device = idxs.device

        # 1. Create key_padding_mask for the idxs part based on semantic_valid_lengths
        # This mask is True for padded semantic tokens, False otherwise (valid semantic, reconstruction).
        key_padding_mask_for_idxs = torch.zeros_like(idxs, dtype=torch.bool) # Default to False (not masked)

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
        xs = None 
        # semantic_content_has_ended_flags[i] is True if self.num_vq was sampled in semantic phase for batch item i
        semantic_content_has_ended_flags = torch.zeros(B, dtype=torch.bool, device=clip_feature.device)

        for _ in range(self.block_size): 
            current_len = xs.shape[1] if xs is not None else 0
            
            if current_len >= self.block_size:
                break

            current_input_for_transformer = xs if xs is not None else []
            
            logits = self.forward(current_input_for_transformer, clip_feature)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1) # (B, V)

            next_step_tokens_for_batch = torch.zeros(B, dtype=torch.long, device=clip_feature.device)

            if current_len < self.semantic_len: # Processing semantic part
                candidate_tokens_this_step = None # Shape (B,)
                if if_categorial:
                    dist = Categorical(probs)
                    candidate_tokens_this_step = dist.sample() 
                else:
                    _, topk_tokens = torch.topk(probs, k=1, dim=-1) 
                    candidate_tokens_this_step = topk_tokens.squeeze(-1)

                for i in range(B):
                    if semantic_content_has_ended_flags[i]:
                        # Semantic content for this item ended, force padding (self.num_vq)
                        next_step_tokens_for_batch[i] = self.num_vq + 1
                    else:
                        token_for_item_i = candidate_tokens_this_step[i]
                        if token_for_item_i == self.num_vq: # First time self.num_vq in semantic part
                            semantic_content_has_ended_flags[i] = True 
                        next_step_tokens_for_batch[i] = token_for_item_i
                
                idx_sampled_token_tensor = next_step_tokens_for_batch.unsqueeze(-1) # (B,1)

            else: # Processing reconstruction part (current_len >= self.semantic_len)
                # self.num_vq is true stop. Follows sample_original_backup style for batch stop.
                idx_sampled_token_tensor = None 
                if if_categorial:
                    dist = Categorical(probs)
                    idx_val_batch = dist.sample() # (B,)
                    # Simplified batch stop: if first item stops, all stop.
                    # A robust batch solution would mask completed sequences.
                    if idx_val_batch[0] == self.num_vq: 
                        break 
                    idx_sampled_token_tensor = idx_val_batch.unsqueeze(-1)
                else:
                    _, idx_val_topk_batch = torch.topk(probs, k=1, dim=-1) # (B,1)
                    if idx_val_topk_batch[0,0] == self.num_vq: 
                        break 
                    idx_sampled_token_tensor = idx_val_topk_batch
            
            if idx_sampled_token_tensor is None: # Should only happen if loop broke in recon phase
                break

            if xs is None:
                xs = idx_sampled_token_tensor
            else:
                xs = torch.cat((xs, idx_sampled_token_tensor), dim=1)
                
        if xs is None: 
            return torch.empty((B, 0), dtype=torch.long, device=clip_feature.device)
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

    def sample_interleaved(self, clip_feature, if_categorial=False):
        xs = None
        has_sampled_reconstruction_token = False
        for k_loop_iter in range(self.block_size - 1):
            current_input_for_transformer = xs if xs is not None else []
            logits = self.forward(current_input_for_transformer, clip_feature)
            logits = logits[:, -1, :]
            probs_orig = F.softmax(logits, dim=-1)
            temp_probs = probs_orig.clone()

            k_pattern = k_loop_iter % 5

            if k_pattern == 0: # Semantic token
                temp_probs[:, self.SEMANTIC_TOKEN_END_IDX + 1 : self.num_vq] = 0 # Mask out recon and separator
                temp_probs[:, self.num_vq] = 0 # Also mask out stop if no recon token yet and in semantic part of pattern
            else: # Reconstruction token
                temp_probs[:, :self.RECONSTRUCTION_TOKEN_START_IDX] = 0 # Mask out semantic and separator
                # For stop token, if no reconstruction token has been sampled yet, mask stop.
                # Otherwise, use its original probability.
                if not has_sampled_reconstruction_token:
                    temp_probs[:, self.num_vq] = 0
                # else: stop_token_prob is already in temp_probs from clone

            batch_sums = temp_probs.sum(dim=-1, keepdim=True)
            for i_batch in range(temp_probs.shape[0]):
                if batch_sums[i_batch] < 1e-9:
                    temp_probs[i_batch, :] = 0
                    if k_pattern == 0: # Semantic fallback
                        # Try original probabilities for semantic range first
                        allowed_orig_semantic_probs = probs_orig[i_batch, :self.SEMANTIC_TOKEN_END_IDX + 1]
                        if allowed_orig_semantic_probs.sum() > 1e-9:
                            temp_probs[i_batch, :self.SEMANTIC_TOKEN_END_IDX + 1] = allowed_orig_semantic_probs
                            # if not has_sampled_reconstruction_token: temp_probs[i_batch, self.num_vq] = 0 # re-mask stop if needed (already done above)
                        else: # Absolute fallback for semantic if original also zero
                            temp_probs[i_batch, 0] = 1.0 # Force token 0
                    else: # Reconstruction fallback
                        allowed_orig_recon_probs = probs_orig[i_batch, self.RECONSTRUCTION_TOKEN_START_IDX : self.num_vq]
                        if allowed_orig_recon_probs.sum() > 1e-9:
                            temp_probs[i_batch, self.RECONSTRUCTION_TOKEN_START_IDX : self.num_vq] = allowed_orig_recon_probs
                            if has_sampled_reconstruction_token: # if recon allowed, stop can also be considered
                                temp_probs[i_batch, self.num_vq] = probs_orig[i_batch, self.num_vq]
                            # else: stop remains 0 if no recon sampled yet
                        else: # Absolute fallback for reconstruction if original also zero
                            temp_probs[i_batch, self.RECONSTRUCTION_TOKEN_START_IDX] = 1.0 # Force first recon token
            
            batch_sums = temp_probs.sum(dim=-1, keepdim=True)
            probs_masked = temp_probs / (batch_sums + 1e-9)

            idx_sampled_token_val = None
            if if_categorial:
                dist = Categorical(probs_masked)
                idx_val = dist.sample()
                if idx_val == self.num_vq:
                    break
                idx_sampled_token_val = idx_val.unsqueeze(-1)
            else:
                _, idx_val_topk = torch.topk(probs_masked, k=1, dim=-1)
                if idx_val_topk[0,0] == self.num_vq:
                    break
                idx_sampled_token_val = idx_val_topk

            if xs is None:
                xs = idx_sampled_token_val
            else:
                xs = torch.cat((xs, idx_sampled_token_val), dim=1)
            
            # Update flag if a reconstruction token was sampled
            sampled_token_id = idx_sampled_token_val[0,0].item()
            if self.RECONSTRUCTION_TOKEN_START_IDX <= sampled_token_id < self.num_vq:
                has_sampled_reconstruction_token = True
        
        if xs is None:
            return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
        return xs
    
    def sample_semantic(self, clip_feature, if_categorial=False):
        xs = None
        current_sampling_phase = "semantic" 
        sampled_at_least_one_reconstruction_token = False
        
        # Loop to generate up to block_size - 1 tokens
        # (condition clip_feature takes one spot in the transformer's view, 
        # effectively allowing block_size-1 generated tokens to form a sequence of self.block_size with condition)
        for k_loop_iter in range(self.block_size -1): 
            current_input_for_transformer = xs if xs is not None else []
            
            logits = self.forward(current_input_for_transformer, clip_feature)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)

            # Mask probabilities based on the current sampling phase
            # self.num_vq is the ID of the stop token. Max actual data token ID is self.num_vq - 1.
            temp_probs = probs.clone()

            if current_sampling_phase == "semantic":
                # Allow [0, SEMANTIC_TOKEN_END_IDX] or SEPARATOR_TOKEN_IDX
                # Block [RECONSTRUCTION_TOKEN_START_IDX, self.num_vq-1 (max_token_id)]
                temp_probs[:, self.RECONSTRUCTION_TOKEN_START_IDX : self.num_vq] = 0 
                # Also block explicit stop if in semantic phase before separator
                temp_probs[:, self.num_vq] = 0 
                    # If all semantic token probabilities are extremely low, force SEPARATOR_TOKEN_IDX
                if torch.sum(temp_probs[:, :self.SEPARATOR_TOKEN_IDX]) < 1e-9 :
                    temp_probs[:, :self.SEPARATOR_TOKEN_IDX+1] = 0 # Zero out semantic and separator initially
                    temp_probs[:, self.SEPARATOR_TOKEN_IDX] = 1.0 # Force separator

            elif current_sampling_phase == "reconstruction":
                # Allow [RECONSTRUCTION_TOKEN_START_IDX, self.num_vq-1] or STOP_TOKEN (self.num_vq)
                # Block all semantic tokens and the separator token
                temp_probs[:, :self.RECONSTRUCTION_TOKEN_START_IDX] = 0
            
            # Renormalize probabilities
            batch_sums = temp_probs.sum(dim=-1, keepdim=True)
            # Handle cases where all valid probabilities become zero after masking
            for i_batch in range(temp_probs.shape[0]):
                if batch_sums[i_batch] < 1e-9: # If sum is too small
                    if current_sampling_phase == "semantic":
                        # Fallback: force SEPARATOR_TOKEN_IDX
                        temp_probs[i_batch, :] = 0
                        temp_probs[i_batch, self.SEPARATOR_TOKEN_IDX] = 1.0
                    else: # reconstruction phase
                        # Fallback: force STOP_TOKEN_ID
                        temp_probs[i_batch, :] = 0
                        temp_probs[i_batch, self.num_vq] = 1.0 
            # Re-calculate sums after potential fallback
            batch_sums = temp_probs.sum(dim=-1, keepdim=True)
            # Avoid division by zero if somehow still all zeros (should be caught by fallback)
            probs_masked = temp_probs / (batch_sums + 1e-9)


            # Sample token from masked probabilities
            idx_sampled_token_val = None
            if if_categorial:
                dist = Categorical(probs_masked)
                idx_val = dist.sample()
                if idx_val == self.num_vq: # Stop token
                    break
                idx_sampled_token_val = idx_val.unsqueeze(-1)
            else:
                _, idx_val_topk = torch.topk(probs_masked, k=1, dim=-1)
                if idx_val_topk[0,0] == self.num_vq: # Stop token
                    break
                idx_sampled_token_val = idx_val_topk

            # Append to sequence
            if xs is None:
                xs = idx_sampled_token_val
            else:
                xs = torch.cat((xs, idx_sampled_token_val), dim=1)

            # Update phase and flags
            sampled_token_id = idx_sampled_token_val[0,0].item()
            if current_sampling_phase == "semantic" and sampled_token_id == self.SEPARATOR_TOKEN_IDX:
                current_sampling_phase = "reconstruction"
            elif current_sampling_phase == "reconstruction":
                if self.RECONSTRUCTION_TOKEN_START_IDX <= sampled_token_id < self.num_vq:
                    sampled_at_least_one_reconstruction_token = True
        
        # After loop, ensure reconstruction token if needed
        if current_sampling_phase == "reconstruction" and not sampled_at_least_one_reconstruction_token:
            # If sequence is empty or only separator, add a reconstruction token
            if xs is None or xs.shape[1] == 0 :
                xs = torch.tensor([[self.RECONSTRUCTION_TOKEN_START_IDX]], dtype=torch.long, device=clip_feature.device)
            elif xs[0,-1].item() == self.SEPARATOR_TOKEN_IDX: # Ends with separator
                # Append a reconstruction token if space allows (block_size-1 max generated tokens)
                if xs.shape[1] < (self.block_size -1):
                    xs = torch.cat((xs, torch.tensor([[self.RECONSTRUCTION_TOKEN_START_IDX]], dtype=torch.long, device=clip_feature.device)), dim=1)
                else: # No space, replace separator
                    xs[0,-1] = self.RECONSTRUCTION_TOKEN_START_IDX
            elif xs[0,-1].item() != self.num_vq : # Last token is not STOP and not SEPARATOR
                xs[0,-1] = self.RECONSTRUCTION_TOKEN_START_IDX # Replace last token

        if xs is None:
            return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
        return xs

    def sample(self, clip_feature, if_categorial=False):
        if self.semantic_interleaved_flag:
            return self.sample_interleaved(clip_feature, if_categorial)
        elif self.semantic_flag:
            return self.sample_semantic(clip_feature, if_categorial)
        elif self.dual_head_flag:
            return self.sample_dual_head(clip_feature, if_categorial)
        else:
            return self.sample_original_backup(clip_feature, if_categorial)

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
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
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
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
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


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
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

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
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
        # self.ln_f = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
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

        

