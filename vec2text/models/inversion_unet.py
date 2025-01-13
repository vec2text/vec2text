import logging
from typing import Optional, Tuple

import torch

from vec2text.models.config import InversionConfig
from vec2text.models import InversionModel

logger = logging.getLogger(__name__)


class UNetTransform(torch.nn.Module):
    def __init__(
            self,
            src_dim: int,
            target_dim: int,
        ):
        super().__init__()
        
        import diffusers
        self.base = diffusers.UNet2DModel(
            sample_size=(32, 32), # the target image resolution
            in_channels=1, # the number of input channels, 3 for RGB images
            out_channels=1, # the number of output channels
            layers_per_block=2, # how many ResNet layers to use per UNet block
            block_out_channels=(128,128,256,256,512,512), # the numbe of output channels for eaxh UNet block
            down_block_types=(
                "DownBlock2D", # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D", # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", # a regular ResNet upsampling block
                "AttnUpBlock2D", # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.internal_dim = src_dim
        self.reshape_factor = 32
        assert self.internal_dim % self.reshape_factor == 0
        self.in_projection = torch.nn.Linear(src_dim, 1024)
        self.out_projection = torch.nn.Linear(1024, target_dim)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = self.in_projection(x).contiguous()
        x = x[:, None, None, :].reshape(-1, 1, 32, 32) # repeat embedding into an image
        output = self.base(x, timestep=0).contiguous()
        z = output.sample.reshape(batch_size, 1024)
        return self.out_projection(z).contiguous()



class InversionModelUnet(InversionModel):
    """A class of model that conditions on embeddings from a pre-trained sentence embedding model
    to decode text autoregressively.
    """
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.unet_transform = UNetTransform(
            self.embedder_dim, self.embedder_dim
        )

    def embed_and_project(
        self,
        embedder_input_ids: Optional[torch.Tensor],
        embedder_attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert not ((embedder_input_ids is None) and (frozen_embeddings is None))
        if frozen_embeddings is not None:
            embeddings = frozen_embeddings
            assert len(embeddings.shape) == 2  # batch by d
        elif self.embedder_no_grad:
            with torch.no_grad():
                embeddings = self.call_embedding_model(
                    input_ids=embedder_input_ids,
                    attention_mask=embedder_attention_mask,
                )
        else:
            embeddings = self.call_embedding_model(
                input_ids=embedder_input_ids,
                attention_mask=embedder_attention_mask,
            )
        
        # pass embeddings through unet
        embeddings = self.unet_transform(embeddings)

        if self.embedding_transform_strategy == "repeat":
            if embeddings.dtype != self.dtype:
                embeddings = embeddings.to(self.dtype)
            repeated_embeddings = self.embedding_transform(embeddings)
            # linear outputs a big embedding, reshape into a sequence of regular size embeddings.
            embeddings = repeated_embeddings.reshape(
                (*repeated_embeddings.shape[:-1], self.num_repeat_tokens, -1)
            )
        elif self.embedding_transform_strategy == "nearest_neighbors":
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError(
                f"unknown embedding transformation strategy {self.embedding_transform_strategy}"
            )
        attention_mask = torch.ones(
            (embeddings.shape[0], embeddings.shape[1]), device=embeddings.device
        )
        return embeddings, attention_mask
