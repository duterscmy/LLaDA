import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                        cfg_scale=0., remasking='low_confidence', mask_id=126336, beam_size=2, log=False, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        beam_size: Beam size for beam search.
    '''
    import json
    temperature=0.5
    print("======PBS, temperature: {:.1f}====".format(temperature))
    # 初始化beam: [(sequence, cumulative_log_prob, block_progress, records)]
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    beam = [(x.clone(), 0.0, 0, [])]  # (sequence, cumulative_log_prob, current_block, records)
    
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    if log:
        print(f"=== Beam Search Generation Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps_per_block}, Beam size: {beam_size}")
        print(f"Initial x shape: {x.shape}")
        print(f"Initial x[-128:]: {x[0, -128:].cpu().numpy().tolist()}")
        print(f"Initial mask count: {(x == mask_id).sum().item()}")
        

    for global_step in range(steps):
        if log:
            print(f"=== Global Step {global_step + 1}/{steps} ===")
        new_beam_candidates = []
        
        # 处理当前beam中的每个候选序列
        for beam_idx, (seq, cumulative_log_prob, current_block, records) in enumerate(beam):
            if log:
                print(f"--- Processing Beam {beam_idx + 1}/{len(beam)} ---")
                print(f"Current cumulative log prob: {cumulative_log_prob:.4f}")
                print(f"Current block progress: {current_block}/{num_blocks}")
                print(f"Current sequence last 128 tokens: {seq[0, -128:].cpu().numpy().tolist()}")
            
            # 确定当前处理的block
            block_start = prompt.shape[1] + current_block * block_length
            block_end = prompt.shape[1] + (current_block + 1) * block_length
            
            # 获取当前block的mask信息
            block_mask_index = (seq[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            if log:
                print(f"Processing block range: [{block_start}, {block_end})")
                print(f"Num transfer tokens for this step: {num_transfer_tokens[0, global_step % steps_per_block]}")
            
            # 前向传播
            mask_index = (seq == mask_id)
            if cfg_scale > 0.:
                un_seq = seq.clone()
                un_seq[prompt_index] = mask_id
                seq_ = torch.cat([seq, un_seq], dim=0)
                logits = model(seq_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(seq).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # 限制在当前block及之前
            x0_p[:, prompt.shape[1] + (current_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, seq)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # (1) 打印所有mask position的confidence
            mask_positions = torch.where(mask_index[0])[0]
            mask_confidence = confidence[0, mask_positions]
            if log:
                print(f"(1) All mask positions: {mask_positions.cpu().numpy().tolist()}")
                print(f"    Mask confidences: {mask_confidence.cpu().float().detach().numpy().tolist()}")
            
            # 选择要unmask的位置
            selected_positions = []
            selected_confidences = []
            transfer_indexs = []
            for j in range(confidence.shape[0]):
                k = beam_size
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    for tmp_select_index in select_index.cpu().numpy().tolist():
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        transfer_index[j, tmp_select_index] = True
                        # (2) 记录选择了哪个position进行unmask，置信度是多少
                        selected_positions.extend([tmp_select_index])
                        selected_confidences.extend([confidence[j, tmp_select_index].item()])
                        transfer_indexs.extend([transfer_index])
            
            assert len(selected_positions) == len(selected_confidences)
            assert len(selected_positions) == len(transfer_indexs)

            if log:
                print(f"(2) Selected positions: {selected_positions}")
                print(f"    Selected confidences: {selected_confidences}")
            
            # 生成新的候选序列
            if len(selected_positions) > 0:
                for selected_position, selected_confidence, transfer_index in zip(selected_positions, selected_confidences, transfer_indexs):
                    new_seq = seq.clone()
                    new_seq[transfer_index] = x0[transfer_index]
                    
                    # 获取预测的token
                    token = x0[0, selected_position].item()
                    
                    # 计算新的累计概率
                    selected_probs = torch.tensor(selected_confidence, device=seq.device)
                    new_log_prob = cumulative_log_prob + selected_probs.sum().item()
                    
                    # 更新block进度
                    new_current_block = current_block
                    if new_current_block < num_blocks - 1:
                        current_block_mask = (new_seq[:, block_start:block_end] == mask_id)
                        if not current_block_mask.any():
                            new_current_block += 1
                            if log:
                                print(f"    Block {current_block} completed, moving to block {new_current_block}")
                    
                    # 创建新的records列表并添加当前解码记录
                    new_records = records.copy()
                    new_records.append({
                        "step": global_step + 1,
                        "position": selected_position,
                        "confidence": selected_confidence,
                        "token_id": token
                    })
                    
                    new_beam_candidates.append((new_seq, new_log_prob, new_current_block, new_records))
                    
                    # (3) 打印unmask后的序列
                    if log:
                        print(f"(3) New sequence last 128 tokens: {new_seq[0, -128:].cpu().numpy().tolist()}")
                        print(f"    New cumulative log prob: {new_log_prob:.4f}")
                    
                    # 打印变化
                    changed_positions = torch.where(seq[0] != new_seq[0])[0]
                    if len(changed_positions) > 0:
                        if log:
                            print(f"    Changed positions: {changed_positions.cpu().numpy().tolist()}")
                            print(f"    Before values: {seq[0, changed_positions].cpu().numpy().tolist()}")
                            print(f"    After values: {new_seq[0, changed_positions].cpu().numpy().tolist()}")
            else:
                # 如果没有选择任何位置，保持原序列
                new_beam_candidates.append((seq, cumulative_log_prob, current_block, records))
                if log:
                    print(f"(3) No positions selected, keeping original sequence")
                    
            if log:
                print(f"--- End Beam {beam_idx + 1}/{len(beam)} ---")
                
        
        # 如果没有生成新的候选，提前结束
        if not new_beam_candidates:
            if log:
                print("No new beam candidates generated, early stopping")
            break
        
        # 全局排序，选择top beam_size个序列
        if log:
            print(f"Total candidates before selection: {len(new_beam_candidates)}")
        new_beam_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 去重
        uniq_new_beam_candidates = []
        seen = set()
        for tensor, float_val, block_progress, records in new_beam_candidates:
            tensor_tuple = tuple(tensor.flatten().cpu().numpy().tolist())
            if tensor_tuple not in seen:
                seen.add(tensor_tuple)
                uniq_new_beam_candidates.append((tensor, float_val, block_progress, records))
        
        if log:
            print(f"Unique candidates after deduplication: {len(uniq_new_beam_candidates)}")
        
        # 更新beam
        beam = uniq_new_beam_candidates[:beam_size]
        
        # 打印当前beam状态
        best_seq, best_score, best_block, best_records = beam[0]
        if log:
            print(f"Current beam scores: {[score for _, score, _, _ in beam]}")
            print(f"Best sequence last 128 tokens: {best_seq[0, -128:].cpu().numpy().tolist()}")
            print(f"Best sequence score: {best_score:.4f}")
            print(f"Remaining mask count: {(best_seq == mask_id).sum().item()}")
            

    # 选择beam中最好的序列作为最终结果
    if beam:
        best_sequence, best_score, _, best_records = beam[0]
        if log:
            print(f"=== Beam Search Generation Complete ===")
            print(f"Final sequence last 128 tokens: {best_sequence[0, -128:].cpu().numpy().tolist()}")
            print(f"Final sequence score: {best_score:.4f}")
            print(f"Final mask count: {(best_sequence == mask_id).sum().item()}")
            print(f"Total decoding records: {len(best_records)}")
            
            # 输出records的简单统计
            if best_records:
                steps_used = max(r["step"] for r in best_records)
                avg_confidence = sum(r["confidence"] for r in best_records) / len(best_records)
                print(f"Steps used: {steps_used}")
                print(f"Average confidence: {avg_confidence:.4f}")
                
                # 输出前几个解码记录作为示例
                print(f"\n=== Top 5 Decoding Records ===")
                for i, record in enumerate(best_records[:5]):
                    print(f"Step {record['step']}: position {record['position']}, "
                          f"token {record['token_id']}, confidence {record['confidence']:.4f}")
    else:
        best_sequence = x
        best_records = []
        if log:
            print(f"=== Beam Search Generation Complete (No valid sequences) ===")

    # 输出top1 beam的records作为JSON
    import json
    print(json.dumps(best_records))
    print(len(best_records))
    
    return best_sequence


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/lus/lfs1aip2/projects/public/u6er/mingyu/models/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Let $O(0,0), A(\tfrac{1}{2}, 0),$ and $B(0, \tfrac{\sqrt{3}}{2})$ be points in the coordinate plane. Let $\mathcal{F}$ be the family of segments $\overline{PQ}$ of unit length lying in the first quadrant with $P$ on the $x$-axis and $Q$ on the $y$-axis. There is a unique point $C$ on $\overline{AB}$, distinct from $A$ and $B$, that does not belong to any segment from $\mathcal{F}$ other than $\overline{AB}$. Then $OC^2 = \tfrac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p + q$.",
               "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=1024, gen_length=1024, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
