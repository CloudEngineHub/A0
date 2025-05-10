import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.dit.model import DiT


class DiTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(DiTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = DiT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size
        )
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=state_token_dim,    # state + state mask (indicator)
            out_features=hidden_size
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim


        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)
                # projector = projector.to(torch.bfloat16) ##

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond,
                           ): #ctrl_freqs,action_mask, state_traj
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = lang_cond.device
        dtype = lang_cond.dtype
        noisy_action = torch.randn(
            size=(lang_cond.shape[0], self.pred_horizon , self.action_dim ), 
            dtype=dtype, device=device)
        # action_mask = action_mask.expand(-1, self.pred_horizon, -1)
        
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
      
            action_traj = self.state_adaptor(noisy_action)
            
            # Predict the model output
            model_output = self.model(action_traj, #ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(lang_cond.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions

        noisy_action = noisy_action 

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                      action_gt, 
                    ) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  


        # Sample noise that we'll add to the actions
        noise = torch.randn(
            size=action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, #1000
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # expand dim 1 of action_gt to T+1
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        

        lang_cond, img_cond, noisy_action = self.adapt_conditions(
            lang_tokens, img_tokens, noisy_action)
        
        
        # Predict the denoised result
        pred = self.model(noisy_action, #ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        target_real = target
        pred_coor = pred


        weights = torch.tensor([0.6, 0.1, 0.1, 0.1,0.1], dtype=torch.float32, device=pred_coor.device)
        weights = weights.unsqueeze(0).unsqueeze(2)

        l1_loss = F.l1_loss(pred_coor, target_real)

        l1_loss_weighted = l1_loss * weights

        # Calculate the unweighted square difference
        squared_diff = F.mse_loss(pred_coor, target_real, ) # reduction='none'

        weighted_squared_diff = squared_diff * weights

        loss = weighted_squared_diff.mean() 

        return {'loss':loss,'mse_loss':squared_diff.sum(),'l1_loss':l1_loss.sum(),'l1_loss_weighted':l1_loss_weighted.mean()}
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, 
                       ): 
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''

        lang_cond = self.lang_adaptor(lang_tokens)
        img_cond = self.img_adaptor(img_tokens)

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond,
        )
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
