# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch, pdb
import torch.nn as nn
from ...lpips import LPIPS
import torch.nn.functional as F
from .gan_loss import softplus_g_loss
from ..engine.misc import get_world_size
from .discriminator_dino_interleave import DinoDisc as DINODiscriminator
from .discriminator_stylegan import Discriminator as StyleGANDiscriminator
from .discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from .distill_loss import DistillLoss

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0., use_warmup=False):
    if global_step < threshold:
        if use_warmup and threshold > 0:
            alpha = global_step / float(threshold)
            return value + alpha * (weight - value)
        else:
            return value
    return weight

class VQLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, stage=1, config=None , use_warmup=False,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", "dinogan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating", "softplus_g_loss"]

        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == 'dinogan':  # the DINOv1-S discriminator from the paper
            norm_type = 'bn'
            self.discriminator = DINODiscriminator(norm_type=norm_type)
            if get_world_size() > 1:
                self.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss  # discriminator loss (disc_loss): use the hinge loss (stabilizes training, avoids vanishing gradients)
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating", "softplus_g_loss"]

        # gen_adv_loss
        if gen_adv_loss == "hinge":  # generator adversarial loss (gen_adv_loss): matched to the discriminator loss so the generator and discriminator loss functions are compatible
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        elif gen_adv_loss == 'softplus_g_loss':
            self.gen_adv_loss = softplus_g_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()  # perceptual loss (perceptual_loss): initialize the LPIPS model (the perceptual loss mentioned in Section 3.1 of the paper), measuring the high-level visual similarity between the reconstruction and the original image.
        self.perceptual_weight = perceptual_weight
        self.cos = nn.CosineSimilarity(dim=2, eps = 1e-6)  # the core contribution of Section 3.3 of the paper, the "semantic reconstruction objective": align the tokenizer output with the features of the vision foundation model to preserve semantic fidelity.
        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss  # reconstruction loss (rec_loss): supports L1/L2; the paper uses the L2 loss (corresponding to F.mse_loss in the code)
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight
        self.DistillLoss = DistillLoss(config, stage)
        self.use_warmup = use_warmup

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        # addresses the paper's need to "balance the reconstruction loss against the adversarial loss", preventing an overly strong adversarial loss from degrading reconstruction quality
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, codebook_loss, inputs, outputs, optimizer_idx, global_step, hidden_states=None, grid_thw=None, last_layer=None, 
                logger=None, log_every=100):
        reconstructions, latent, dinov2 = outputs  # shapes: [16, 3, 336, 336], [16, 576, 1024], [16, 576, 1024]
        #* Generator update
        if optimizer_idx == 0:
            #* Reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous()) + F.mse_loss(inputs.contiguous(), reconstructions.contiguous())

            #* Perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)

            #* Discriminator loss
            logits_fake, _ = self.discriminator(reconstructions.contiguous(), None)

            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start, use_warmup=self.use_warmup)
            
            distill_loss = self.DistillLoss(hidden_states, grid_thw, latent, dinov2)
            loss = self.rec_weight * rec_loss + \
                self.perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2] + distill_loss
            
            #* Log losses
            if (global_step % log_every == 0) & (logger is not None):

                rec_loss = self.rec_weight * rec_loss
                p_loss = self.perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                logger.info(f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                            f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                            f"codebook_usage: {codebook_loss[3]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, "
                            f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}")
            return loss, distill_loss

        #* Discriminator update
        if optimizer_idx == 1:
            
            logits_fake, logits_real = self.discriminator(reconstructions.contiguous().detach(), inputs.contiguous().detach())
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)
            if (global_step % log_every == 0) & (logger is not None):
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}")
            return d_adversarial_loss