import torch.nn as nn
import torch as th
from stable_baselines3.common.torch_layers import MlpExtractor

class CustomMLPExtractor(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn, device="auto"):
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get('pi', [64])
            vf_layers = net_arch.get('vf', [64])
            
            if pi_layers:
                max_pi = max(pi_layers)
                enhanced_pi = pi_layers + [max(max_pi * 2, 128)]
            else:
                enhanced_pi = [64, 128]

            if vf_layers:
                enhanced_vf = vf_layers
                if not any(layer >= 256 for layer in vf_layers):
                    enhanced_vf.append(256)
            else:
                enhanced_vf = [64, 256]

            enhanced_net_arch = {
                'pi': enhanced_pi,
                'vf': enhanced_vf
            }
        else:
            enhanced_net_arch = dict(pi=[64, 128], vf=[64, 256])

        super().__init__(feature_dim, [enhanced_net_arch], activation_fn, device)

        if hasattr(self, 'shared_net') and len(self.shared_net) > 0:
            shared_output_dim = self.latent_dim_pi if self.shared_net else feature_dim
            self.use_residual = (self.latent_dim_pi == shared_output_dim)
        else:
            self.use_residual = (self.latent_dim_pi == feature_dim)

    def forward(self, features: th.Tensor):
        shared_latent = self.shared_net(features)
        policy_latent = self.policy_net(shared_latent)
        value_latent = self.value_net(shared_latent)

        if self.use_residual:
            policy_latent = policy_latent + 0.1 * shared_latent

        return policy_latent, value_latent