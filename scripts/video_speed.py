# python scripts/video_speed.py --dir "/Users/gmh/oasis/code/aff/A0/videos" --speed 2.0

import argparse
from pathlib import Path
import subprocess


def change_video_speed(input_path, speed, output_path):
    print(f"Processing {input_path} at {speed}x speed...")

    # Check if the input video has an audio stream
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(input_path),
    ]
    result = subprocess.run(
        probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    has_audio = bool(result.stdout.strip())

    speed_filter = f"setpts={1/speed}*PTS"
    text_filter = f"drawtext=text='{speed}x':x=w-tw-10:y=h-th-10:fontsize=24:fontcolor=white:borderw=2"

    if has_audio:
        if speed < 0.5 or speed > 2.0:
            raise ValueError("ffmpeg atempo only supports speed between 0.5 and 2.0")
        audio_speed = f"atempo={speed}"
        filter_complex = f"[0:v]{speed_filter},{text_filter}[v];[0:a]{audio_speed}[a]"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            str(output_path),
        ]
    else:
        filter_complex = f"[0:v]{speed_filter},{text_filter}[v]"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            str(output_path),
        ]

    subprocess.run(cmd, check=True)
    print(f"Saved to {output_path}\n")


def process_directory(directory, speed, out_dir=""):
    input_dir = Path(directory)
    output_root = (
        Path(out_dir)
        if out_dir
        else input_dir.parent / f"{input_dir.name}_x{str(speed).replace('.', '_')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for file_path in input_dir.iterdir():
        if file_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            output_path = output_root / file_path.name
            change_video_speed(str(file_path), speed, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change speed of all videos in a directory."
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to the dirƒectory with videos"
    )
    parser.add_argument(
        "--speed",
        type=float,
        required=True,
        help="Speed factor (e.g. 1.5 for 1.5x speed)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to input_dir_x{speed}",
    )
    args = parser.parse_args()

    process_directory(args.dir, args.speed, args.out_dir)
