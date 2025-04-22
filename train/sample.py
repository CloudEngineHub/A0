from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def log_sample_res(
    text_encoder, vision_encoder, rdt, args, 
    accelerator, weight_dtype, dataset_id2name, dataloader, logger
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()
    
    loss_for_log = defaultdict(float)
    loss_counter = defaultdict(int)
    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break
        
        data_indices = batch["data_indices"]

        images = batch["images"].to(dtype=weight_dtype)
        # states = batch["states"].to(dtype=weight_dtype)
        # We only use the last state as input

        actions = batch["actions"].to(dtype=weight_dtype)

            
        batch_size, _, C, H, W = images.shape
        image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
        image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
        
        lang_attn_mask = batch["lang_attn_mask"]
        text_embeds = batch["lang_embeds"].to(dtype=weight_dtype) \
            if args.precomp_lang_embed \
            else text_encoder(
                input_ids=batch["input_ids"],
                attention_mask=lang_attn_mask
            )["last_hidden_state"].detach()

        pred_actions = rdt.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,

        )
        

        weights = torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float32, device=pred_actions.device) # 定义你的权重张量
        weights = weights.unsqueeze(0).unsqueeze(2)
        
        loss = F.mse_loss(pred_actions, actions,).float() # reduction='none'
        
        loss_weighted = loss * weights
        
        l1_loss = F.l1_loss(pred_actions, actions)
        l1_loss_weighted = l1_loss * weights

        first_point_diff = torch.abs(pred_actions[:, 0, :] - actions[:, 0, :])
        
        loss_for_log['test_mse_loss'] = loss.sum().detach().item()
        loss_for_log['test_mse_loss_weighted'] = loss_weighted.mean().detach().item()
        loss_for_log['test_l1_loss'] = l1_loss.sum().detach().item()
        loss_for_log['test_l1_loss_weighted'] = l1_loss_weighted.mean().detach().item()
        loss_for_log['test_l1_loss_first'] = first_point_diff.mean().detach().item()


    rdt.train()
    torch.cuda.empty_cache()

    return dict(loss_for_log)