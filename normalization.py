"""
normalization.py — shared utility
===================================
Single source of truth for all geometric normalization functions.
Imported by step2_train.py, step3_test_desktop.py, and any other
script that needs to process keypoints.

These functions MUST remain identical to the TypeScript versions in
inferenceHelper.ts — any change here must be mirrored there.
"""

import numpy as np


SEQUENCE_LENGTH = 30
N_FEATURES      = 225   # 63 + 63 + 99
POSE_START      = 126   # pose features start at index 126
LH_END          = 63
RH_START        = 63
RH_END          = 126


def normalize_hand(hand_flat: np.ndarray) -> np.ndarray:
    """
    Wrist-relative + scale-normalized.
    Input / output: (63,)  float32

    Steps:
      1. Subtract wrist (landmark 0) from all 21 landmarks
      2. Divide by distance from wrist to middle-finger MCP (landmark 9)
         → invariant to hand size and camera distance
    """
    pts   = hand_flat.reshape(21, 3).copy().astype(np.float32)
    wrist = pts[0].copy()
    pts  -= wrist
    scale = float(np.linalg.norm(pts[9]))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()


def normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
    """
    Shoulder-midpoint-relative + shoulder-width-normalized.
    Input / output: (99,)  float32

    Steps:
      1. Subtract midpoint of left (11) and right (12) shoulder
      2. Divide by shoulder width
         → invariant to position in frame and body size
    """
    pts    = pose_flat.reshape(33, 3).copy().astype(np.float32)
    l_sh   = pts[11].copy()
    r_sh   = pts[12].copy()
    mid    = (l_sh + r_sh) / 2.0
    pts   -= mid
    width  = float(np.linalg.norm(l_sh - r_sh))
    if width > 1e-6:
        pts /= width
    return pts.flatten()


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply hand + pose normalization to a single frame vector.
    Input / output: (225,)
    """
    lh   = normalize_hand(frame[:LH_END].copy())
    rh   = normalize_hand(frame[RH_START:RH_END].copy())
    pose = normalize_pose(frame[POSE_START:].copy())
    return np.concatenate([lh, rh, pose]).astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Apply normalize_frame to every frame in a sequence.
    Input / output: (30, 225)
    """
    out = np.zeros_like(seq, dtype=np.float32)
    for i, frame in enumerate(seq):
        out[i] = normalize_frame(frame)
    return out
