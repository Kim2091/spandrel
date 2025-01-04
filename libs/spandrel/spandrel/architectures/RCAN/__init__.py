from spandrel.util import KeyCondition, get_pixelshuffle_params, get_seq_len

from ...__helpers.model_descriptor import Architecture, ImageModelDescriptor, StateDict
from .__arch.rcan_arch import RCAN


class RCANArch(Architecture[RCAN]):
    def __init__(self):
        super().__init__(
            id="RCAN",
            detect=KeyCondition.has_all(
                "head.0.weight",
                "tail.1.weight",
                "body.0.body.0.body.0.weight",
                "body.0.body.0.body.3.conv_du.0.weight",
            ),
        )

    def load(self, state_dict: StateDict) -> ImageModelDescriptor[RCAN]:
        n_resgroups = get_seq_len(state_dict, "body") - 1
        n_resblocks = get_seq_len(state_dict, "body.0.body") - 1
        n_colors = state_dict["head.0.weight"].shape[1]
        rgb_range = 255  # undetectable
        kernel_size = state_dict["head.0.weight"].shape[-1]
        scale, n_feats = get_pixelshuffle_params(state_dict, "tail.0")
        norm = "sub_mean.weight" in state_dict
        reduction = (
            n_feats // state_dict["body.0.body.0.body.3.conv_du.0.weight"].shape[0]
        )
        res_scale = 1  # undetectable
        act_mode = "relu"  # undetectable

        model = RCAN(
            scale=scale,
            n_resgroups=n_resgroups,
            n_resblocks=n_resblocks,
            n_colors=n_colors,
            rgb_range=rgb_range,
            norm=norm,
            kernel_size=kernel_size,
            n_feats=n_feats,
            reduction=reduction,
            res_scale=res_scale,
            act_mode=act_mode,
        )

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if scale == 1 else "SR",
            tags=[f"{n_feats}nf", f"{n_resgroups}nrg", f"{n_resblocks}nrb"],
            supports_half=True,
            supports_bfloat16=True,
            scale=scale,
            input_channels=n_colors,
            output_channels=n_colors,
        )


__all__ = ["RCANArch", "RCAN"]