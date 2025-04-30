# python scripts/video_speed.py --dir videos --speed 2.0 --format gif

import argparse
from pathlib import Path
import subprocess


def change_video_speed(input_path, speed, output_path, output_format="mp4"):
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

    if output_format == "gif":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            f"{speed_filter},{text_filter},fps=15,scale=480:'trunc(ih*480/iw/2)*2':flags=lanczos",
            str(output_path),
        ]
    else:
        if has_audio and 0.5 <= speed <= 2.0:
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


def process_directory(directory, speed, out_dir="", output_format="mp4"):
    input_dir = Path(directory)
    output_root = (
        Path(out_dir)
        if out_dir
        else input_dir.parent / f"{input_dir.name}_x{str(speed).replace('.', '_')}_{output_format}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for file_path in input_dir.iterdir():
        if file_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            output_filename = file_path.stem + f".{output_format}"
            output_path = output_root / output_filename
            change_video_speed(str(file_path), speed, output_path, output_format)


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
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "gif"],
        help="Output format: mp4 or gif"
    )
    args = parser.parse_args()

    process_directory(args.dir, args.speed, args.out_dir, args.format)
