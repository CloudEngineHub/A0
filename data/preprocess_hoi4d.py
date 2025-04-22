"""
HOI4D dataset: https://hoi4d.github.io/
HOI4D dataset processing script. Get the waypoints from origin dataset via image segmentation and event annotation.
Functionality: Extract mask information from annotations, compute centroids, overlay masks onto the original video frames, and preserve the original directory structure
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import re

import pandas as pd

import csv
import imageio.v2 as imageio

from concurrent.futures import ProcessPoolExecutor, as_completed

# Mapping from category labels to chosen label indices
category_choice_label = {
    'C1': 1, 'C2': [1], 'C3': 1, 'C4': 1, 'C5': 1, 'C6': 3,
    'C7': 1, 'C8': [3, 5], 'C9': [1], 'C11': [1], 'C12': 1,
    'C13': 1, 'C14': 3, 'C17': 3, 'C18': 4, 'C20': 1
}


def get_color_map(N=256):
    """
    Return the RGB color for each label index.
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = np.array([r, g, b])
    return cmap


def get_video_fps(video_path):
    """
    Get the frames per second (FPS) and total frame count of a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def load_event_mapping(annotation_dir, fps=30):
    """
    Load event information from color.json:
      - Returns a dict mapping frame_idx to event_name for visualization
      - Returns a list of aligned event frame ranges for metadata
    """
    json_path = os.path.join(annotation_dir, "action", "color.json")
    frame_event_map = {}
    events_frame_aligned = []

    if not os.path.exists(json_path):
        return frame_event_map, events_frame_aligned

    with open(json_path, "r") as f:
        color_data = json.load(f)

    # Detect JSON format
    if "events" in color_data:
        events = color_data.get("events", [])
        duration = color_data.get("info", {}).get("duration",
                     color_data.get("info", {}).get("Duration", None))
    elif "markResult" in color_data and "marks" in color_data["markResult"]:
        events = color_data["markResult"]["marks"]
        duration = color_data.get("info", {}).get("Duration",
                     color_data.get("info", {}).get("duration", None))
    else:
        return frame_event_map, events_frame_aligned

    scale = 1.0
    if duration == 10:
        scale = 2.0

    for event in events:
        # Choose appropriate time fields
        if "startTime" in event and "endTime" in event:
            start_time = event["startTime"]
            end_time = event["endTime"]
        elif "hdTimeStart" in event and "hdTimeEnd" in event:
            start_time = event["hdTimeStart"]
            end_time = event["hdTimeEnd"]
        else:
            continue

        start_frame = int(start_time * fps * scale)
        end_frame = int(end_time * fps * scale)

        # Map each frame to its event
        for fidx in range(start_frame, end_frame + 1):
            frame_event_map[fidx] = event["event"]

        # Record aligned frame ranges
        events_frame_aligned.append({
            "id": event["id"],
            "event": event["event"],
            "start_frame": start_frame,
            "end_frame": end_frame
        })

    return frame_event_map, events_frame_aligned


def get_merged_center_from_mask(mask, label_indices, color_map):
    """
    Compute the centroid of the merged mask regions for given label indices.
    """
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    # Convert BGR to RGB for matching
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    merged_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)

    for label_idx in label_indices:
        target_color = np.array(color_map[label_idx])
        match = np.all(mask_rgb == target_color, axis=-1)
        merged_mask |= match

    coords = np.column_stack(np.where(merged_mask))
    if coords.size == 0:
        return None
    cy, cx = coords.mean(axis=0)
    return (int(cx), int(cy))


def get_merged_center_from_mask_cv2(mask, label_indices, color_map):
    """
    Compute centroid using contours and moments for given label indices.
    """
    if isinstance(label_indices, int):
        label_indices = [label_indices]

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    binary_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)

    for label_idx in label_indices:
        target_color = np.array(color_map[label_idx])
        match = np.all(mask_rgb == target_color, axis=-1)
        binary_mask[match] = 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (int(cx), int(cy))


def extract_category_from_path(image_dir):
    """
    Extract category string (e.g., 'C6') and number from a directory path.
    """
    for part in image_dir.split("/"):
        if part.startswith("C") and part[1:].isdigit():
            return part, int(part[1:])
    return None, None


def get_task_name(image_dir, csv_path="/mnt/data/xurongtao/HOI4D-Instructions/definitions/task/task_definitions.csv"):
    """
    Extract category (Cx) and task (Tx) from the path and retrieve the task name from CSV.
    """
    category_match = re.search(r"(C\d+)", image_dir)
    task_match = re.search(r"(T\d+)", image_dir)

    if not category_match or not task_match:
        raise ValueError("Path does not contain valid category (Cx) or task (Tx)")

    category_id = category_match.group(1)
    task_column = task_match.group(1)
    df = pd.read_csv(csv_path)
    row = df.loc[df["Category ID"] == category_id]
    if row.empty:
        raise ValueError(f"Category {category_id} not found in CSV.")

    task_name = row.iloc[0][task_column]
    if pd.isna(task_name) or task_name == "N/A":
        raise ValueError(f"Task {task_column} not defined for category {category_id}.")
    return task_name


def get_label_count_for_category(category_id, label_csv_path="/mnt/data/xurongtao/HOI4D-Instructions/definitions/motion segmentation/label.csv"):
    """
    Count non-empty labels (Label 1~6) for a given category from label.csv.
    Returns (count, category_name).
    """
    with open(label_csv_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Category ID'] == category_id:
                count = sum(bool(row.get(f"Lable {i}", '').strip()) for i in range(1, 7))
                return count, row['Category']
    return 0, None


def get_label_name(category_id: str, label_idx: int, label_csv_path="/mnt/data/xurongtao/HOI4D-Instructions/definitions/motion segmentation/label.csv") -> str:
    """
    Retrieve the label name for a given category and label index.
    """
    if not (1 <= label_idx <= 6):
        raise ValueError("label_idx must be in range 1~6")

    with open(label_csv_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Category ID'] == category_id:
                return row.get(f"Lable {label_idx}", "").strip() or "Unknown"
    return "Unknown"


def process_video_frames(image_dir, annotation_dir, output_dir, color_map):
    """
    Process frames for a single video segment:
      - Read frames, overlay mask centroids and events, write to output video, and save metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")

    origin_video_path = os.path.join(image_dir, "image.mp4")
    fps_detected, total_frames = get_video_fps(origin_video_path)
    print(f"Detected video FPS: {fps_detected}, total frames: {total_frames}")

    task_name = get_task_name(image_dir)
    frame_event_map, events_frame_aligned = load_event_mapping(annotation_dir, fps=fps_detected)

    mapped_frames = [f for f in range(total_frames) if f in frame_event_map]
    print(f"Total frames: {total_frames}, frames with events: {len(mapped_frames)}, without events: {total_frames-len(mapped_frames)}")

    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))

    Cx, x = extract_category_from_path(image_dir)
    N, cate_name = get_label_count_for_category(Cx)
    labels = category_choice_label[Cx]
    # Special handling for certain categories
    if x == 4:
        if 'door' in task_name:
            labels = 1
        elif 'drawer' in task_name:
            labels = 4
    labels_t = labels[0] if isinstance(labels, list) else labels

    label_name0 = get_label_name(Cx, labels_t)
    label_name = label_name0 if cate_name == label_name0 else f"{label_name0} of {cate_name}"
    if x in [9, 11]:
        label_name = cate_name if x == 9 else label_name0

    relative_image_dir = os.path.relpath(image_dir, start=DATASET_ROOT)
    metadata = {"image_dir": relative_image_dir,"label_name": label_name,"task_name": task_name,"frames": []}

    output_video_path = os.path.join(output_dir, "masked.mp4")
    writer = imageio.get_writer(output_video_path, fps=fps_detected, macro_block_size=1)

    for image_name in image_files:
        frame_str = Path(image_name).stem
        frame_path = os.path.join(image_dir, image_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        mask_path = os.path.join(annotation_dir, "2Dseg", "mask", f"{frame_str}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(annotation_dir, "2Dseg", "shift_mask", f"{frame_str}.png")

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            labels_list = labels if isinstance(labels, list) else [labels]
            center = (get_merged_center_from_mask_cv2 if x == 8 else get_merged_center_from_mask)(mask, labels_list, color_map)

            overlay = frame.copy()
            if center:
                matched_color = tuple(int(c) for c in color_map[labels_t])
                color_bgr = tuple(int(c) for c in matched_color[::-1])
                cv2.circle(overlay, center, 8, color_bgr, -1)
                cv2.putText(overlay, label_name, (center[0]+15, center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

            current_event = frame_event_map.get(int(frame_str), None)
            metadata["frames"].append({"frame_index": int(frame_str), "center": center, "event": current_event})

            mask_vis = cv2.imread(mask_path)
            overlay = cv2.addWeighted(overlay, 0.7, mask_vis, 0.3, 0)

            if current_event:
                cv2.putText(overlay, f"Event: {current_event}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(overlay, f"Task: {task_name}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        else:
            print(f"Mask not found: {mask_path}")
            overlay = frame

        resized_overlay = cv2.resize(overlay, (overlay.shape[1]//3, overlay.shape[0]//3), interpolation=cv2.INTER_AREA)
        writer.append_data(cv2.cvtColor(resized_overlay, cv2.COLOR_BGR2RGB))

    writer.close()
    print(f"Saved video to: {output_video_path}")
    metadata["events_frame"] = events_frame_aligned
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")


def process_dataset(annotations_root, dataset_root, output_root):
    """
    Process the entire dataset:
      - Traverse annotation directories, find video segments, and process them in parallel.
    """
    color_map = get_color_map()
    tasks = []
    for root, dirs, files in os.walk(annotations_root):
        if "2Dseg" in dirs:
            rel = os.path.relpath(root, annotations_root)
            img_dir = os.path.join(dataset_root, rel, "align_rgb")
            out_dir = os.path.join(output_root, rel)
            if os.path.isdir(img_dir):
                tasks.append((img_dir, root, out_dir))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_video_frames, img, ann, out, color_map) for img, ann, out in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing video segments"):
            pass


if __name__ == "__main__":
    ANNOTATIONS_ROOT = "/mnt/data/xurongtao/datasets/datasets--JianZhangAI--hoi4d/HOI4D_annotations"
    DATASET_ROOT = "/mnt/data/xurongtao/datasets/datasets--JianZhangAI--hoi4d/HOI4D_release"
    OUTPUT_ROOT = "/mnt/data/xurongtao/datasets/datasets--JianZhangAI--hoi4d/HOI4D_converted"
    process_dataset(ANNOTATIONS_ROOT, DATASET_ROOT, OUTPUT_ROOT)
