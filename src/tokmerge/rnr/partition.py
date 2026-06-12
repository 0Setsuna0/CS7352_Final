from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VisualLayout:
    """Latent video-token grid used by CogVideoX attention.

    Tokens are flattened in frame-major order: ``t * H * W + h * W + w``.
    """

    frames: int
    height: int
    width: int

    @property
    def num_tokens(self) -> int:
        return self.frames * self.height * self.width

    def validate(self, n_tokens: int) -> None:
        expected = self.num_tokens
        if n_tokens != expected:
            raise AssertionError(f"visual token count {n_tokens} != layout tokens {expected}")


def normalize_stride(dst_stride: tuple[int, int, int] | list[int] | int) -> tuple[int, int, int]:
    if isinstance(dst_stride, int):
        stride = (dst_stride, dst_stride, dst_stride)
    else:
        if len(dst_stride) != 3:
            raise ValueError(f"dst_stride must have 3 values, got {dst_stride!r}")
        stride = tuple(int(v) for v in dst_stride)
    if any(v <= 0 for v in stride):
        raise ValueError(f"dst_stride values must be positive, got {stride!r}")
    return stride


_PARTITION_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _strided_destination_indices(
    layout: VisualLayout,
    stride: tuple[int, int, int],
    device: torch.device,
    protect_first_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(layout.num_tokens, device=device)
    hw = layout.height * layout.width
    t = idx // hw
    rem = idx % hw
    h = rem // layout.width
    w = rem % layout.width

    dst_mask = (t % stride[0] == 0) & (h % stride[1] == 0) & (w % stride[2] == 0)
    if dst_mask.sum().item() == 0:
        dst_mask[0] = True

    src_mask = ~dst_mask
    if protect_first_frame:
        src_mask = src_mask & (t != 0)

    src_idx = src_mask.nonzero(as_tuple=False).squeeze(-1)
    dst_idx = dst_mask.nonzero(as_tuple=False).squeeze(-1)
    return src_idx, dst_idx


def _random_chunk_destination_indices(
    layout: VisualLayout,
    stride: tuple[int, int, int],
    device: torch.device,
    protect_first_frame: bool,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style spatiotemporal split.

    Each non-overlapping chunk contributes one randomly selected destination
    token; all other tokens become source candidates, except protected tokens.
    """

    frames, height, width = layout.frames, layout.height, layout.width
    chunk_f, chunk_h, chunk_w = stride
    num_chunk_f, mod_f = divmod(frames, chunk_f)
    num_chunk_h, mod_h = divmod(height, chunk_h)
    num_chunk_w, mod_w = divmod(width, chunk_w)

    if num_chunk_f == 0 or num_chunk_h == 0 or num_chunk_w == 0:
        return _strided_destination_indices(layout, stride, device, protect_first_frame)

    chunk_tokens = chunk_f * chunk_h * chunk_w
    num_dst_tokens = num_chunk_f * num_chunk_h * num_chunk_w
    dst_local = torch.randint(
        high=chunk_tokens,
        size=(num_chunk_f, num_chunk_h, num_chunk_w, 1),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )

    indicators = torch.zeros(
        num_chunk_f,
        num_chunk_h,
        num_chunk_w,
        chunk_tokens,
        device=device,
        dtype=torch.int64,
    )
    indicators.scatter_(-1, dst_local, torch.ones_like(dst_local))
    indicators = (
        indicators.reshape(num_chunk_f, num_chunk_h, num_chunk_w, chunk_f, chunk_h, chunk_w)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(frames - mod_f, height - mod_h, width - mod_w)
    )

    if mod_f > 0 or mod_h > 0 or mod_w > 0:
        pad_left_f = pad_left_h = pad_left_w = 0
        if mod_f > 0:
            pad_left_f = int(torch.randint(mod_f, (1,), device=device, generator=generator).item())
        if mod_h > 0:
            pad_left_h = int(torch.randint(mod_h, (1,), device=device, generator=generator).item())
        if mod_w > 0:
            pad_left_w = int(torch.randint(mod_w, (1,), device=device, generator=generator).item())
        pad = (
            pad_left_w,
            mod_w - pad_left_w,
            pad_left_h,
            mod_h - pad_left_h,
            pad_left_f,
            mod_f - pad_left_f,
        )
        indicators = F.pad(indicators, pad, mode="constant", value=0)

    flat_idx = indicators.reshape(-1).argsort(descending=True, stable=True)
    dst_idx = flat_idx[:num_dst_tokens]
    src_idx = flat_idx[num_dst_tokens:]

    if protect_first_frame:
        t = src_idx // (layout.height * layout.width)
        src_idx = src_idx[t != 0]

    return src_idx, dst_idx


def source_destination_indices(
    layout: VisualLayout,
    dst_stride: tuple[int, int, int] | list[int] | int,
    device: torch.device,
    protect_first_frame: bool = True,
    partition_mode: str = "random_chunk",
    generator: torch.Generator | None = None,
    cache_key_extra: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source and destination candidate token indices.

    Destination tokens are the regular strided lattice; all other visual tokens
    are source candidates. Protected frame-0 tokens are kept out of the source
    set, so they can still serve as destinations but are never discarded.
    """

    stride = normalize_stride(dst_stride)
    mode = partition_mode.lower()
    key = (layout.frames, layout.height, layout.width, stride, protect_first_frame, mode, str(device), cache_key_extra)
    if generator is None:
        cached = _PARTITION_CACHE.get(key)
        if cached is not None:
            return cached

    if mode == "strided":
        src_idx, dst_idx = _strided_destination_indices(layout, stride, device, protect_first_frame)
    elif mode == "random_chunk":
        src_idx, dst_idx = _random_chunk_destination_indices(layout, stride, device, protect_first_frame, generator)
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode!r}")

    if generator is None:
        _PARTITION_CACHE[key] = (src_idx, dst_idx)
    return src_idx, dst_idx
