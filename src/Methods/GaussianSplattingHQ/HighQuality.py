"""
GaussianSplattingHQ/HighQuality.py

High-quality variant of 3D Gaussian Splatting that exposes several previously
hardcoded quality parameters as YAML-configurable options, and adds an optional
VGG perceptual loss term.

This is a purely additive subclass — no upstream files are modified.
Activate by setting METHOD_TYPE: GaussianSplattingHQ in your config file.
"""

import torch

import Framework
from Methods.Base.utils import training_callback
from Methods.GaussianSplatting.Model import Gaussians, GaussianSplattingModel
from Methods.GaussianSplatting.Loss import GaussianSplattingLoss
from Methods.GaussianSplatting.Renderer import GaussianSplattingRenderer
from Methods.GaussianSplatting.Trainer import GaussianSplattingTrainer


# ---------------------------------------------------------------------------
# Model — overrides split() and densify_and_prune() to accept configurable
# factors instead of hardcoded constants (1.6 and 0.1 respectively).
# ---------------------------------------------------------------------------

class HighQualityGaussians(Gaussians):
    """Gaussians subclass with configurable split scale divisor and prune factor."""

    def split(self, grads: torch.Tensor, grad_threshold: float, scale_divisor: float = 1.6) -> None:
        """Densify by splitting Gaussians that satisfy the gradient condition.
        
        Args:
            grads: Accumulated gradient norms.
            grad_threshold: Minimum gradient magnitude to trigger a split.
            scale_divisor: Scale of each child Gaussian relative to the parent
                (child = parent / scale_divisor). Smaller values produce denser,
                finer-grained children. Default 1.6 matches the original 3DGS paper.
        """
        from Cameras.utils import quaternion_to_rotation_matrix
        from Optim.adam_utils import extend_param_groups, prune_param_groups

        n_init_points = self.get_positions.shape[0]
        padded_grad = torch.zeros(n_init_points, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask &= torch.max(self.get_scales, dim=1).values > self.percent_dense * self.training_cameras_extent

        stds = self.get_scales[selected_pts_mask].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_rotation_matrix(self._rotations[selected_pts_mask]).repeat(2, 1, 1)
        new_positions = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions[selected_pts_mask].repeat(2, 1)
        new_scales = self.inverse_scaling_activation(self.get_scales[selected_pts_mask].repeat(2, 1) / scale_divisor)
        new_rotations = self._rotations[selected_pts_mask].repeat(2, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)
        new_opacities = self._opacities[selected_pts_mask].repeat(2, 1)

        self.densification_postfix(new_positions, new_features_dc, new_features_rest, new_opacities, new_scales, new_rotations)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum().item(), device='cuda', dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_prune(
        self,
        grad_threshold: float,
        min_opacity: float,
        prune_large_gaussians: bool,
        large_gaussians_factor: float = 0.1,
        scale_divisor: float = 1.6,
    ) -> None:
        """Densify the point cloud and prune points that are not visible or too large.

        Args:
            grad_threshold: Gradient threshold for triggering split/duplicate.
            min_opacity: Gaussians below this opacity are pruned. Lower values
                preserve more semi-transparent splats.
            prune_large_gaussians: Whether to prune Gaussians that exceed the
                size threshold.
            large_gaussians_factor: Size threshold multiplier relative to
                training_cameras_extent. Higher values allow larger Gaussians
                to persist. Default 0.1 matches the original 3DGS paper.
            scale_divisor: Passed through to split(). Default 1.6 matches the
                original 3DGS paper.
        """
        grads = self.densification_gradient_accum / self.n_observations.clamp_min(1)

        self.duplicate(grads, grad_threshold)
        self.split(grads, grad_threshold, scale_divisor)

        prune_mask = self.get_opacities.flatten() < min_opacity
        if prune_large_gaussians:
            prune_mask |= self.get_scales.max(dim=1).values > large_gaussians_factor * self.training_cameras_extent
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


class HighQualityModel(GaussianSplattingModel):
    """GaussianSplattingModel subclass that uses HighQualityGaussians internally."""

    def build(self) -> 'HighQualityModel':
        """Builds the model using HighQualityGaussians."""
        pretrained = self.num_iterations_trained > 0
        self.gaussians = HighQualityGaussians(self.SH_DEGREE, pretrained)
        return self


# ---------------------------------------------------------------------------
# Renderer — thin subclass that widens the accepted model type list to include
# HighQualityModel. BaseRenderer uses an exact type() check (not isinstance),
# so we must explicitly add the subclass to the valid_model_types list.
# ---------------------------------------------------------------------------

class HighQualityRenderer(GaussianSplattingRenderer):
    """GaussianSplattingRenderer that also accepts HighQualityModel instances."""

    def __init__(self, model) -> None:
        # Pass both accepted types so BaseRenderer's exact-type guard is satisfied
        from Methods.Base.Renderer import BaseRenderer
        BaseRenderer.__init__(self, model, [GaussianSplattingModel, HighQualityModel])
        import Framework
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('GaussianSplatting renderer not implemented in CPU mode')





# ---------------------------------------------------------------------------
# Loss — extends the base GS loss with an optional VGG perceptual term.
# ---------------------------------------------------------------------------

class HighQualityLoss(GaussianSplattingLoss):
    """GaussianSplattingLoss subclass that optionally adds a VGG perceptual loss.

    When loss_config.LAMBDA_VGG == 0.0 (the default), VGGLoss is never imported
    or instantiated — there is zero overhead compared to the base loss.
    """

    def __init__(self, loss_config: Framework.ConfigParameterList) -> None:
        super().__init__(loss_config)
        self._vgg_weight = float(loss_config.LAMBDA_VGG) if hasattr(loss_config, 'LAMBDA_VGG') else 0.0
        self._vgg_loss = None
        if self._vgg_weight > 0.0:
            from Optim.Losses.VGG import VGGLoss
            self._vgg_loss = VGGLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss: L1 + DSSIM (+ optional VGG)."""
        loss = super().forward(input, target)
        if self._vgg_loss is not None:
            loss = loss + self._vgg_weight * self._vgg_loss(input, target)
        return loss


# ---------------------------------------------------------------------------
# Trainer — adds new quality config params and overrides the two callbacks
# that previously used hardcoded constants.
# ---------------------------------------------------------------------------

@Framework.Configurable.configure(
    PRUNE_MIN_OPACITY=0.005,
    PRUNE_LARGE_GAUSSIANS_FACTOR=0.1,
    SH_DEGREE_INCREASE_INTERVAL=1000,
    SPLIT_SCALE_DIVISOR=1.6,
    LOSS=Framework.ConfigParameterList(
        LAMBDA_L1=0.8,
        LAMBDA_DSSIM=0.2,
        LAMBDA_VGG=0.0,
    ),
)
class HighQualityTrainer(GaussianSplattingTrainer):
    """GaussianSplattingTrainer subclass exposing additional quality parameters.

    New config parameters (all defaults preserve original 3DGS behavior):

        PRUNE_MIN_OPACITY (float, default 0.005):
            Gaussians with opacity below this are pruned. Lower values keep
            more semi-transparent splats, potentially improving translucent detail.

        PRUNE_LARGE_GAUSSIANS_FACTOR (float, default 0.1):
            Gaussians whose largest scale exceeds (factor * cameras_extent) are
            pruned. Increase to allow larger Gaussians to persist in big scenes.

        SH_DEGREE_INCREASE_INTERVAL (int, default 1000):
            How many iterations between each SH degree ramp-up step, and at
            which iteration the ramp starts. Increasing this delays when
            higher-frequency view-dependent color kicks in.

        SPLIT_SCALE_DIVISOR (float, default 1.6):
            When a Gaussian is split into two children, each child's scale =
            parent_scale / SPLIT_SCALE_DIVISOR. Lower values produce smaller,
            denser children at the cost of higher primitive count.

        LOSS.LAMBDA_VGG (float, default 0.0):
            Weight of the VGG perceptual loss term. Set to e.g. 0.1 to add a
            perceptual quality signal on top of L1 + DSSIM. When 0.0, VGGLoss
            is never instantiated and has zero runtime or memory cost.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Replace the base-class loss with HighQualityLoss (which may include VGG)
        self.loss = HighQualityLoss(loss_config=self.LOSS)

    @training_callback(
        priority=110,
        start_iteration='SH_DEGREE_INCREASE_INTERVAL',
        iteration_stride='SH_DEGREE_INCREASE_INTERVAL',
    )
    @torch.no_grad()
    def increase_sh_degree(self, *_) -> None:
        """Increase used SH coefficients at the configured interval."""
        self.model.gaussians.increase_used_sh_degree()

    @training_callback(
        priority=90,
        start_iteration='DENSIFY_START_ITERATION',
        end_iteration='DENSIFY_END_ITERATION',
        iteration_stride='DENSIFICATION_INTERVAL',
    )
    @torch.no_grad()
    def densify(self, iteration: int, _) -> None:
        """Apply densification using the configured quality parameters."""
        if iteration == self.DENSIFY_START_ITERATION or iteration == self.DENSIFY_END_ITERATION:
            return
        self.model.gaussians.densify_and_prune(
            self.DENSIFY_GRAD_THRESHOLD,
            self.PRUNE_MIN_OPACITY,
            iteration > self.OPACITY_RESET_INTERVAL,
            self.PRUNE_LARGE_GAUSSIANS_FACTOR,
            self.SPLIT_SCALE_DIVISOR,
        )
