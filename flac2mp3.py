#!/usr/bin/env python3
"""
FLAC to MP3 Converter - Optimized for LLM transcription
Drag and drop a .flac file onto this script (or the .bat file)

Settings: 48kbps mono, 22kHz - gives ~75-80% compression
"""

import subprocess
import sys
from pathlib import Path


def get_ffmpeg_path() -> Path:
    """Find ffmpeg in same dir, ffmpeg subdir, or PATH."""
    script_dir = Path(__file__).parent

    # Check ffmpeg subfolder first
    ffmpeg_subdir = script_dir / "ffmpeg" / "ffmpeg.exe"
    if ffmpeg_subdir.exists():
        return ffmpeg_subdir

    # Check same folder
    ffmpeg_local = script_dir / "ffmpeg.exe"
    if ffmpeg_local.exists():
        return ffmpeg_local

    # Fall back to PATH
    return Path("ffmpeg")


def convert(input_file: str) -> None:
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return

    if input_path.suffix.lower() != ".flac":
        print(f"Error: Expected .flac file, got {input_path.suffix}")
        return

    output_path = input_path.with_suffix(".mp3")
    ffmpeg = get_ffmpeg_path()

    print(f"Converting: {input_path.name}")
    print(f"Output: {output_path.name}")
    print()

    cmd = [
        str(ffmpeg),
        "-i", str(input_path),
        "-ac", "1",           # mono
        "-ar", "22050",       # 22kHz sample rate
        "-ab", "48k",         # 48 kbps bitrate
        "-y",                 # overwrite output
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: ffmpeg not found!")
        print(f"Expected at: {ffmpeg}")
        return
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg failed with code {e.returncode}")
        return

    # Show file size comparison
    in_size = input_path.stat().st_size
    out_size = output_path.stat().st_size
    reduction = (1 - out_size / in_size) * 100

    print()
    print(f"Done: {output_path}")
    print(f"  {in_size:,} bytes -> {out_size:,} bytes ({reduction:.1f}% reduction)")


def main():
    if len(sys.argv) < 2:
        print("FLAC to MP3 Converter")
        print("=" * 40)
        print("Drag and drop a .flac file onto this script")
        print("Or run: python flac2mp3.py yourfile.flac")
        input("\nPress Enter to exit...")
        return

    # Process all dropped files
    for filepath in sys.argv[1:]:
        convert(filepath)
        print()

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
