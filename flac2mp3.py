#!/usr/bin/env python3
"""Convert FLAC files to 96kbps mono MP3."""

import argparse
import subprocess
import sys
from pathlib import Path


def convert(input_file: str, output_file: str | None = None) -> None:
    input_path = Path(input_file)

    if not input_path.exists():
        sys.exit(f"Error: File not found: {input_file}")

    if input_path.suffix.lower() != ".flac":
        sys.exit(f"Error: Expected .flac file, got {input_path.suffix}")

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = input_path.with_suffix(".mp3")

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-ac", "1",          # mono
        "-ab", "96k",        # 96 kbps bitrate
        "-y",                # overwrite output
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("Error: ffmpeg not found. Install it first.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error: ffmpeg failed with code {e.returncode}")

    # Show file size comparison
    in_size = input_path.stat().st_size
    out_size = output_path.stat().st_size
    reduction = (1 - out_size / in_size) * 100

    print(f"\nDone: {output_path}")
    print(f"  {in_size:,} bytes → {out_size:,} bytes ({reduction:.1f}% reduction)")


def main():
    parser = argparse.ArgumentParser(description="Convert FLAC to 96kbps mono MP3")
    parser.add_argument("input", help="Input .flac file")
    parser.add_argument("-o", "--output", help="Output .mp3 file (default: same name)")
    args = parser.parse_args()

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
