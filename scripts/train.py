"""Run mlx_lm.lora with two fixes, so the loss covers exactly the cleaned text and <|im_end|>.

1. mlx-lm forces enable_thinking=True for every tokenizer that has <think> tokens (Qwen3 included).
   With --mask-prompt the loss would then start before the empty <think>\\n\\n</think>\\n\\n block.
   S1-mini gets that block as part of its prompt, so the loss must start after it.
2. mlx-lm's default_loss also scores the first padding token (id 0, "!") after each row. The smoke
   model then answered "!!!!..." to requests that came without think:false, and the reported loss
   included a target that is never generated.
"""
import functools

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import lora
from mlx_lm.tokenizer_utils import TokenizerWrapper


def patch_tokenizer():
    TokenizerWrapper.has_thinking = property(lambda self: False)


def completion_loss(model, batch, lengths):
    """mlx_lm.tuner.trainer.default_loss with `steps < length` instead of `steps <= length`."""
    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps < lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / ntoks, ntoks


if __name__ == "__main__":
    patch_tokenizer()
    lora.train = functools.partial(lora.train, loss=completion_loss)
    lora.main()
