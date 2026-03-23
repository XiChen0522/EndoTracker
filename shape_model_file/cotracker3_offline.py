# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc

# ====== Shape Trace Logger ======
import logging as _logging
_shape_logger = _logging.getLogger("shape_trace")
_shape_logger.setLevel(_logging.DEBUG)
_shape_logger.propagate = False
if not _shape_logger.handlers:
    _fh = _logging.FileHandler("cotracker3_shape_trace.log", mode="w", encoding="utf-8")
    _fh.setFormatter(_logging.Formatter("%(message)s"))
    _shape_logger.addHandler(_fh)
def _slog(msg):
    _shape_logger.info(msg)

_first_iter_done = False
# ================================

torch.manual_seed(0)


class CoTrackerThreeOffline(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTrackerThreeOffline, self).__init__(**args)

    def forward(
        self,
        video,
        queries,
        iters=4,
        is_train=False,
        add_space_attn=True,
        fmaps_chunk_size=200,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        global _first_iter_done

        B, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape

        _slog(f"\n{'='*70}")
        _slog(f"[SHAPE] CoTrackerThreeOffline.forward()")
        _slog(f"  video: ({B}, {T}, {C}, {H}, {W})")
        _slog(f"  queries: ({B}, {N}, {queries.shape[2]})")
        _slog(f"  stride={self.stride}, iters={iters}, corr_levels={self.corr_levels}, corr_radius={self.corr_radius}")
        _slog(f"  latent_dim={self.latent_dim}, input_dim={self.input_dim}")
        _slog(f"  num_virtual_tracks={self.num_virtual_tracks}")
        _slog(f"  model_resolution={self.model_resolution}")
        print(f"[SHAPE TRACE] Output -> cotracker3_shape_trace.log (video={B}x{T}x{C}x{H}x{W}, N={N})")

        assert T >= 1

        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:3]
        queried_coords = queried_coords / self.stride

        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )
        C_ = C
        H4, W4 = H // self.stride, W // self.stride

        _slog(f"  H4={H4}, W4={W4} (feature map resolution)")

        # Compute convolutional features
        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t : t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))

        _slog(f"  fmaps (after fnet, raw): {tuple(fmaps.shape)}")

        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)
        _slog(f"  fmaps (after L2 norm + reshape): {tuple(fmaps.shape)}  = (B, T, latent_dim={self.latent_dim}, H4={H4}, W4={W4})")

        # Build feature pyramid and track features
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        _slog(f"\n  Feature pyramid ({self.corr_levels} levels):")
        for i, fm in enumerate(fmaps_pyramid):
            _slog(f"    level {i}: {tuple(fm.shape)}")

        r = 2 * self.corr_radius + 1
        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        _slog(f"\n  Track features (per pyramid level):")
        for i in range(self.corr_levels):
            _slog(f"    level {i}: track_feat={tuple(track_feat_pyramid[i].shape)}, support={tuple(track_feat_support_pyramid[i].shape)}")

        D_coords = 2

        coord_preds, vis_preds, confidence_preds = [], [], []

        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()

        _slog(f"\n  Initial state:")
        _slog(f"    coords: {tuple(coords.shape)}  = (B, T, N, 2)")
        _slog(f"    vis: {tuple(vis.shape)}  = (B, T, N)")
        _slog(f"    r (support radius): {r}  -> corr_volume per level: ({r}*{r}*{r}*{r}) = {r**4}")

        _first_iter_done = False
        for it in range(iters):
            _detail = not _first_iter_done

            coords = coords.detach()  # B T N 2
            coords_init = coords.view(B * T, N, 2)
            corr_embs = []
            corr_feats = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * T * N, r * r * r * r))
                corr_embs.append(corr_emb)

                if _detail and i == 0:
                    _slog(f"\n  --- iter {it}/{iters} (first iter, full detail) ---")
                    _slog(f"  Correlation computation (level 0 shown):")
                    _slog(f"    corr_feat: {tuple(corr_feat.shape)}  = (B, T, N, {r}, {r}, latent_dim)")
                    _slog(f"    track_feat_support: {tuple(track_feat_support.shape)}  = (B, N, {r}, {r}, latent_dim)")
                    _slog(f"    corr_volume: {tuple(corr_volume.shape)}  = (B, T, N, {r}, {r}, {r}, {r})")
                    _slog(f"    corr_volume reshape: ({B*T*N}, {r**4}) -> corr_mlp -> corr_emb: {tuple(corr_emb.shape)}")

            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, T, N, corr_embs.shape[-1])

            if _detail:
                _slog(f"    corr_embs (all levels cat): {tuple(corr_embs.shape)}  = (B, T, N, {corr_embs.shape[3]}={self.corr_levels}x{corr_emb.shape[-1]})")

            transformer_input = [vis[..., None], confidence[..., None], corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )
            scale = (
                torch.tensor(
                    [self.model_resolution[1], self.model_resolution[0]],
                    device=coords.device,
                )
                / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)

            if _detail:
                _slog(f"    rel_pos_emb: {tuple(rel_pos_emb_input.shape)}  = posenc(4d_rel_coords, deg0~10) -> 4+4*2*10=84")

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            if _detail:
                _slog(f"\n  Token construction (the {self.input_dim}-dim breakdown):")
                _slog(f"    vis:           1")
                _slog(f"    confidence:    1")
                _slog(f"    corr_embs:     {corr_embs.shape[3]}  ({self.corr_levels} levels x {corr_emb.shape[-1]})")
                _slog(f"    rel_pos_emb:   {rel_pos_emb_input.shape[3]}  (posenc of 4d rel coords)")
                _slog(f"    total:         {x.shape[2]}  = input_dim")
                _slog(f"  x (before time_emb): {tuple(x.shape)}  = (B*N={B*N}, T={T}, input_dim)")

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)  # (B N) T D -> B N T D

            if _detail:
                _slog(f"  x (after +time_emb): {tuple(x.shape)}  = (B, N, T, input_dim)")
                _slog(f"\n  >>> updateformer input: {tuple(x.shape)}")

            delta = self.updateformer(
                x,
                add_space_attn=add_space_attn,
            )

            if _detail:
                _slog(f"  <<< updateformer output (delta): {tuple(delta.shape)}  = (B, N, T, output_dim={delta.shape[3]})")

            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence

            coords = coords + delta_coords
            coords_append = coords.clone()
            coords_append[..., :2] = coords_append[..., :2] * float(self.stride)
            coord_preds.append(coords_append)
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

            if _detail:
                _slog(f"  delta_coords: {tuple(delta_coords.shape)}  = (B, T, N, 2)")
                _slog(f"  delta_vis: {tuple(delta_vis.shape)}  = (B, T, N)")
                _slog(f"  delta_confidence: {tuple(delta_confidence.shape)}  = (B, T, N)")
                _slog(f"  updated coords (in pixels): {tuple(coords_append.shape)}")
            else:
                _slog(f"\n  --- iter {it}/{iters} (same shapes, skipped) ---")

            _first_iter_done = True

        _slog(f"\n  Final output:")
        _slog(f"    coords: {tuple(coord_preds[-1].shape)}  = (B, T, N, 2)")
        _slog(f"    vis: {tuple(vis_preds[-1].shape)}  = (B, T, N)")
        _slog(f"    confidence: {tuple(confidence_preds[-1].shape)}  = (B, T, N)")

        if is_train:
            all_coords_predictions.append([coord[..., :2] for coord in coord_preds])
            all_vis_predictions.append(vis_preds)
            all_confidence_predictions.append(confidence_preds)

        if is_train:
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None

        return coord_preds[-1][..., :2], vis_preds[-1], confidence_preds[-1], train_data