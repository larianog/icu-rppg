import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rppg
from scipy.stats import pearsonr


MODELS = [
    # 'ME-chunk.rlap',
    # 'ME-flow.rlap',
    # 'ME-chunk.pure',
    # 'ME-flow.pure',
    # 'PhysMamba.pure',
    # 'PhysMamba.rlap',
    # 'RhythmMamba.rlap',
    # 'RhythmMamba.pure',
    # 'PhysFormer.pure',
    'PhysFormer.rlap',
    # 'TSCAN.rlap',
    # 'TSCAN.pure',
    # 'PhysNet.rlap',
    # 'PhysNet.pure',
    # "EfficientPhys.pure",
    # "EfficientPhys.rlap",
]

VIDEO_NAME = ""
VIDEO_PATH = f"video/{VIDEO_NAME}.avi"
GT_FILE = "ground_truth/sp02wave_L9v2_16-04-13_16-06-12.txt"
SEGMENT_DURATION = 15  # seconds
RESULTS_DIR = "results"


def load_ground_truth(gt_file):
    with open(gt_file, "r") as f:
        _ = f.readline()  # skip PPG line
        gt_hr_line = f.readline()
        gt_ts_line = f.readline()

    gt_hr = np.array([float(x) for x in gt_hr_line.strip().split()], dtype=float)
    gt_ts = np.array([float(x) for x in gt_ts_line.strip().split()], dtype=float)
    return gt_hr, gt_ts


def split_video_into_segments(video_path, segment_duration):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS reported by OpenCV for {video_path}: {fps}")

    frames_per_segment = int(segment_duration * fps)
    if frames_per_segment <= 0:
        cap.release()
        raise RuntimeError("Segment duration produced zero frames per segment.")

    print(f"Video name: {VIDEO_NAME}")
    print(f"Source FPS from OpenCV: {fps:.6f}")
    print(f"Total frames reported by OpenCV: {total_frames}")
    print("Starting video frame splitting")

    segments = []
    start_frame = 0

    while start_frame < total_frames:
        end_frame = min(start_frame + frames_per_segment, total_frames)
        frames_bgr = []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)

        if frames_bgr:
            frames_rgb = np.ascontiguousarray(
                np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr])
            )
            segments.append(
                {
                    "frames_rgb": frames_rgb,
                    "start_time": start_frame / fps,
                    "end_time": (start_frame + len(frames_bgr)) / fps,
                    "num_frames": len(frames_bgr),
                }
            )

        start_frame = end_frame

    cap.release()
    return segments, fps


def chunk_ground_truth_mean(gt_hr, gt_ts, start_time, end_time):
    if len(gt_ts) == 0:
        return np.nan

    mask = (gt_ts >= start_time) & (gt_ts < end_time)
    if np.any(mask):
        return float(np.mean(gt_hr[mask]))

    center_time = (start_time + end_time) / 2.0
    return float(np.interp(center_time, gt_ts, gt_hr))


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return np.nan, np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan

    return pearsonr(x, y)


def run_model_on_segments(model_name, segments, source_fps, gt_hr, gt_ts):
    print(f"\nRunning inference for model: {model_name}")

    hr_array = []
    gt_mean_array = []
    time_array = []
    segment_times = []

    model = rppg.Model(model_name)
    
    # Warmup timing
    print("Initiating warmup")
    _ = model.process_video_tensor(segments[0]["frames_rgb"], fps=source_fps)

    start_total = time.time()
    print("Starting to estimate HR for each video chunk")
    
    for idx, segment in enumerate(segments):
        start_segment = time.time()

        frames_rgb = segment["frames_rgb"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        center_time = (start_time + end_time) / 2.0

        results = model.process_video_tensor(frames_rgb, fps=source_fps)
        estimated_hr = results["hr"] if results is not None else np.nan
        gt_mean_hr = chunk_ground_truth_mean(gt_hr, gt_ts, start_time, end_time)

        print(
            f"Chunk {idx}: "
            f"{start_time:.2f}s to {end_time:.2f}s | "
            f"frames={segment['num_frames']} | "
            f"Estimated HR={estimated_hr} BPM | "
            f"GT mean HR={gt_mean_hr:.2f} BPM"
        )

        hr_array.append(estimated_hr)
        gt_mean_array.append(gt_mean_hr)
        time_array.append(center_time)

        end_segment = time.time()
        segment_times.append(end_segment - start_segment)

    total_time = time.time() - start_total

    hr_array = np.array(hr_array, dtype=float)
    gt_mean_array = np.array(gt_mean_array, dtype=float)
    time_array = np.array(time_array, dtype=float)

    valid = np.isfinite(hr_array) & np.isfinite(gt_mean_array)
    valid_hr = hr_array[valid]
    valid_gt = gt_mean_array[valid]
    valid_time = time_array[valid]

    mean_error_array = np.abs(valid_hr - valid_gt)
    avg_mean_error = float(np.mean(mean_error_array)) if len(mean_error_array) else np.nan
    std_mean_error = float(np.std(mean_error_array)) if len(mean_error_array) else np.nan
    pearson_corr, pearson_p = safe_pearsonr(valid_hr, valid_gt)

    print(f"Total HR estimation time: {total_time:.2f} seconds")
    print("Time taken for each chunk:")
    for i, elapsed in enumerate(segment_times):
        print(f"  Chunk {i}: {elapsed:.2f} seconds")

    print("Valid estimated HR array:", valid_hr)
    print("Valid GT mean HR array:", valid_gt)
    print("Mean error array:", mean_error_array)
    print(f"AVG MEAN ERROR: {avg_mean_error:.2f} ± {std_mean_error:.2f} BPM")
    print(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3g})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_txt_path = os.path.join(
        RESULTS_DIR, f"{model_name}_{VIDEO_NAME}_chunks_result.txt"
    )
    with open(results_txt_path, "w") as f:
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Video Name: {VIDEO_NAME}\n")
        f.write(f"Segment Duration: {SEGMENT_DURATION} seconds\n")
        f.write(f"Source FPS from OpenCV: {source_fps:.6f}\n")
        f.write(f"Total HR estimation time: {total_time:.2f} seconds\n")
        f.write("Time taken for each chunk:\n")
        for i, elapsed in enumerate(segment_times):
            f.write(f"  Chunk {i}: {elapsed:.2f} seconds\n")
        f.write("\n")
        f.write(f"Chunk center times: {valid_time.tolist()}\n")
        f.write(f"Estimated HR: {valid_hr.tolist()}\n")
        f.write(f"GT mean HR per chunk: {valid_gt.tolist()}\n")
        f.write(f"Mean error array: {mean_error_array.tolist()}\n")
        f.write(f"AVG MEAN ERROR: {avg_mean_error:.2f} ± {std_mean_error:.2f} BPM\n")
        f.write(f"Pearson correlation: {pearson_corr:.3f} (p-value: {pearson_p:.3g})\n")

    plot_path = os.path.join(RESULTS_DIR, f"{model_name}_{VIDEO_NAME}_result.png")
    plt.figure(figsize=(10, 5))
    plt.plot(valid_time, valid_hr, marker="o", label="Estimated HR (model)")
    plt.plot(valid_time, valid_gt, marker="x", linestyle="--", label="Ground-truth mean HR")
    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (BPM)")
    plt.title(f"Estimated Heart vs. Ground-Truth mean HR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def main():
    gt_hr, gt_ts = load_ground_truth(GT_FILE)

    segments, source_fps = split_video_into_segments(VIDEO_PATH, SEGMENT_DURATION)

    if not segments:
        raise RuntimeError("No video segments were extracted.")

    for model_name in MODELS:
        run_model_on_segments(model_name, segments, source_fps, gt_hr, gt_ts)


if __name__ == "__main__":
    main()
