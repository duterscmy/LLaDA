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
             cfg_scale=0., remasking='low_confidence', mask_id=126336, log=False, logits_eos_inf=False, confidence_eos_eot_inf=False):
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
    '''
    import json

    # temperature = 0.5
    print("======greedy, temperature: {:.1f}====".format(temperature))
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    # 初始化records列表
    records = []

    if log:
        print(f"=== Generation Start ===")
        print(f"Total blocks: {num_blocks}, Steps per block: {steps}")
        print(f"Initial x shape: {x.shape}")
        print(f"Initial x[-128:]: {x[0, -128:].cpu().float().detach().numpy()}")
        

    for num_block in range(num_blocks):
        if log:
            print(f"=== Block {num_block + 1}/{num_blocks} ===")
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        if log:
            print(f"Block range: [{block_start}, {block_end})")
            print(f"Num transfer tokens: {num_transfer_tokens}")
            

        for i in range(steps):
            if log:
                print(f"--- Step {i + 1}/{steps} (Block {num_block + 1}) ---")
            
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
                
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # (1) 打印所有mask position的confidence
            mask_positions = torch.where(mask_index[0])[0]
            mask_confidence = confidence[0, mask_positions]
            
            if log:
                print(f"Mask positions (indices): {mask_positions.cpu().float().detach().numpy()}")
                print(f"Mask confidences: {mask_confidence.cpu().float().detach().numpy()}")
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            selected_positions = []
            selected_confidences = []
            
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
                
                # (2) 记录选择了哪个position进行unmask，置信度是多少
                selected_positions.extend(select_index.cpu().float().detach().numpy())
                selected_confidences.extend(confidence[j, select_index].cpu().float().detach().numpy())
                
                # 为每个选择的position添加记录
                for pos, conf in zip(select_index, confidence[j, select_index]):
                    pos_int = pos.item()
                    token = x0[j, pos].item()
                    conf_float = conf.item()
                    
                    records.append({
                        "step": i + 1,
                        "block": num_block + 1,
                        "position": pos_int,
                        "confidence": conf_float,
                        "token_id": token
                    })
            
            if log:
                print(f"Selected positions: {selected_positions}")
                print(f"Selected confidences: {selected_confidences}")
            
            # 保存unmask前的状态用于比较
            x_before = x.clone()
            x[transfer_index] = x0[transfer_index]
            
            # (3) 打印unmask后的x[-128:]
            if log:
                print(f"x[-128:] after unmask: {x[0, -128:].cpu().numpy()}")
            
            # 打印unmask的具体变化
            changed_positions = torch.where(x_before[0] != x[0])[0]
            if len(changed_positions) > 0:
                if log:
                    print(f"Changed positions: {changed_positions.cpu().float().detach().numpy()}")
                    print(f"Before values: {x_before[0, changed_positions].cpu().float().detach().numpy()}")
                    print(f"After values: {x[0, changed_positions].cpu().float().detach().numpy()}")
            
              # 空行分隔每个step
    
    if log:
        print(f"=== Generation Complete ===")
        print(f"Total decoding records: {len(records)}")
        
        # 输出records的简单统计
        if records:
            steps_used = max(r["step"] for r in records)
            avg_confidence = sum(r["confidence"] for r in records) / len(records)
            print(f"Steps used: {steps_used}")
            print(f"Average confidence: {avg_confidence:.4f}")
            
            # 输出前几个解码记录作为示例
            print(f"\n=== Top 5 Decoding Records ===")
            for idx, record in enumerate(records[:5]):
                print(f"Step {record['step']} (Block {record['block']}): position {record['position']}, "
                      f"token {record['token_id']}, confidence {record['confidence']:.4f}")

    import json
    # 输出records作为JSON（不带缩进）
    print(json.dumps(records))
    print(len(records))
    return x

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
