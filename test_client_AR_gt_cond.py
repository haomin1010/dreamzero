#!/usr/bin/env python3
"""GT-conditioned test client for AR_droid policy server.

Loads input video frames from debug_image/ and previously-saved *normalized*
actions from the server's debug output directory, then sends both to the
server with the "gt_actions" key so that the server uses
lazy_joint_forward_causal_gt_cond to predict video conditioned on the given
actions.

Usage:
    # Step 1 — run normal inference (server saves debug artifacts automatically):
    python test_client_AR.py --host <host> --port 5000

    # Step 2 — replay with GT-conditioned inference
    #   --actions-dir auto-detects the latest server output dir under checkpoints/
    python test_client_AR_gt_cond.py --host <host> --port 5000

    # Or specify the server output dir explicitly:
    python test_client_AR_gt_cond.py --host <host> --port 5000 \
        --actions-dir /path/to/server/output_dir
"""

import argparse
import glob
import logging
import os
import time
import uuid

import cv2
import numpy as np

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")

CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24

CHECKPOINTS_ROOT = os.path.join(os.path.dirname(__file__), "checkpoints")


def auto_detect_actions_dir() -> str:
    """Find the latest server output directory that contains debug/ artifacts."""
    if not os.path.isdir(CHECKPOINTS_ROOT):
        raise FileNotFoundError(
            f"Checkpoints root not found: {CHECKPOINTS_ROOT}. "
            "Please specify --actions-dir explicitly."
        )
    candidates = []
    for name in os.listdir(CHECKPOINTS_ROOT):
        full = os.path.join(CHECKPOINTS_ROOT, name)
        if not os.path.isdir(full):
            continue
        debug_dir = os.path.join(full, "debug")
        if os.path.isdir(debug_dir):
            candidates.append(full)
            continue
        for sub in os.listdir(full):
            sub_full = os.path.join(full, sub)
            if os.path.isdir(sub_full) and os.path.isdir(os.path.join(sub_full, "debug")):
                candidates.append(sub_full)

    if not candidates:
        raise FileNotFoundError(
            f"No server output directory with debug/ found under {CHECKPOINTS_ROOT}. "
            "Run normal inference first, then retry. "
            "Or specify --actions-dir explicitly."
        )
    best = max(candidates, key=os.path.getmtime)
    logging.info(f"Auto-detected server output dir: {best}")
    return best


def load_all_frames(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current_frame = 23
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks")
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
    gt_actions: np.ndarray | None = None,
) -> dict:
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        if len(frame_indices) == 1:
            selected = selected[0]
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id

    if gt_actions is not None:
        obs["gt_actions"] = gt_actions

    return obs


def load_saved_actions(actions_dir: str) -> list[np.ndarray]:
    """Load **normalized** action arrays saved by the server's debug output.

    Looks for ``normalized_actions.npy`` files under the server's
    ``output_dir/debug/{msg_index}/`` directories.  These are in the model's
    internal format (shape ``[B, action_horizon, model_action_dim]``,
    values in [-1, 1]) and can be passed directly to the GT-cond interface.

    Falls back to client-side ``all_actions.npy`` / ``actions_*.npy`` files
    (roboarena format) if normalized versions are not available.
    """
    # Try server-side normalized actions first (preferred)
    norm_files = sorted(glob.glob(os.path.join(actions_dir, "debug", "*", "normalized_actions.npy")))
    if norm_files:
        actions = [np.load(f) for f in norm_files]
        logging.info(
            f"Loaded {len(actions)} normalized action files from server debug output "
            f"(shape {actions[0].shape})"
        )
        return actions

    # Fallback: client-side stacked actions (roboarena format — may cause shape mismatch)
    all_path = os.path.join(actions_dir, "all_actions.npy")
    if os.path.exists(all_path):
        stacked = np.load(all_path)
        logging.warning(
            f"Using raw roboarena actions from {all_path} (shape {stacked.shape}). "
            "These may not match the model's internal format. "
            "Run a normal inference first so the server saves normalized actions."
        )
        return [stacked[i] for i in range(stacked.shape[0])]

    files = sorted(glob.glob(os.path.join(actions_dir, "actions_*.npy")))
    if not files:
        raise FileNotFoundError(
            f"No saved action files found in {actions_dir}. "
            "Run test_client_AR.py first to generate them, then point "
            "--actions-dir to the server's output_dir (which contains debug/)."
        )
    actions = [np.load(f) for f in files]
    logging.warning(f"Loaded {len(actions)} raw action files (may not match model format)")
    return actions


def _log_action(actions: np.ndarray, dt: float) -> None:
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    logging.info(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )


def test_gt_cond(
    host: str = "localhost",
    port: int = 8000,
    num_chunks: int = 15,
    prompt: str = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    actions_dir: str = "debug_output",
):
    logging.info(f"Connecting to AR_droid server at {host}:{port} (GT-cond mode)...")

    client = WebsocketClientPolicy(host=host, port=port)

    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    server_config = policy_server.PolicyServerConfig(**metadata)
    logging.info(f"Server config: {server_config}")

    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")

    # Load saved actions from a prior normal-inference run
    saved_actions = load_saved_actions(actions_dir)
    logging.info(f"Loaded {len(saved_actions)} saved action arrays")

    # Load video frames
    camera_frames = load_camera_frames()
    total_frames = min(v.shape[0] for v in camera_frames.values())
    chunks = build_frame_schedule(total_frames, num_chunks)

    # Trim to match available actions (initial + chunks)
    max_steps = min(1 + len(chunks), len(saved_actions))
    chunks = chunks[: max_steps - 1]
    logging.info(f"Will run {max_steps} steps (1 initial + {len(chunks)} chunks)")

    # Step 0: initial single frame + GT action from first saved action
    logging.info("=== Initial: frame [0] + GT actions ===")
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id, gt_actions=saved_actions[0])
    t0 = time.time()
    actions = client.infer(obs)
    dt = time.time() - t0
    _log_action(actions, dt)

    # Subsequent chunks
    for chunk_idx, frame_indices in enumerate(chunks):
        action_idx = chunk_idx + 1
        gt_act = saved_actions[action_idx]
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} + GT actions (shape {gt_act.shape}) ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id, gt_actions=gt_act)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        _log_action(actions, dt)

    logging.info("Sending reset to save GT-cond video...")
    client.reset({})
    logging.info("Done (GT-conditioned mode).")


def main():
    parser = argparse.ArgumentParser(
        description="GT-conditioned test: replay saved actions through the world model"
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--num-chunks", type=int, default=15,
        help="Max number of 4-frame chunks (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--actions-dir", default=None,
        help="Server output directory containing debug/ with normalized_actions.npy. "
             "If omitted, auto-detects the latest dir under checkpoints/.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    actions_dir = args.actions_dir
    if actions_dir is None:
        actions_dir = auto_detect_actions_dir()

    test_gt_cond(
        host=args.host,
        port=args.port,
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        actions_dir=actions_dir,
    )


if __name__ == "__main__":
    main()
