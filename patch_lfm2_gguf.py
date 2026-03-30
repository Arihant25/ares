#!/usr/bin/env python3
"""
Patch LFM2-VL GGUF to add missing output_norm.weight tensor.

LFM2 uses weight tying: output_norm = token_embd_norm.
Ollama's llama.cpp expects a separate output_norm tensor.
This script patches the GGUF in-place by:
  1. Shifting the tensor data section forward by 64 bytes
  2. Inserting an output_norm.weight tensor metadata entry
     pointing to the same offset as token_embd_norm.weight
  3. Updating n_tensors from 266 to 267
"""

import struct
import sys
import os

GGUF_MAGIC = b"GGUF"
ALIGNMENT = 32
CHUNK = 4 * 1024 * 1024  # 4MB chunks for the shift


def read_str(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def skip_value(f, vtype):
    if vtype == 8:
        length = struct.unpack("<Q", f.read(8))[0]
        f.read(length)
    elif vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            skip_value(f, atype)
    elif vtype in (0, 1, 7):
        f.read(1)
    elif vtype in (2, 3):
        f.read(2)
    elif vtype in (4, 5, 6):
        f.read(4)
    elif vtype in (10, 11, 12):
        f.read(8)


def scan_gguf(path):
    """Return positions and tensor info needed for patching."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == GGUF_MAGIC, f"Not a GGUF file: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        for _ in range(n_kv):
            read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            skip_value(f, vtype)

        # Read all tensor metadata
        target_name = b"token_embd_norm.weight"
        target_info = None
        for _ in range(n_tensors):
            name_bytes_len = struct.unpack("<Q", f.read(8))[0]
            name_bytes = f.read(name_bytes_len)
            ndims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            if name_bytes == target_name:
                target_info = (ndims, dims, dtype, offset)

        tensor_meta_end = f.tell()

    assert target_info is not None, "token_embd_norm.weight not found in GGUF"

    # Calculate current and new data section starts
    old_data_start = tensor_meta_end
    if old_data_start % ALIGNMENT != 0:
        old_data_start += ALIGNMENT - (old_data_start % ALIGNMENT)

    # New tensor entry for output_norm.weight
    new_name = b"output_norm.weight"
    new_entry = (
        struct.pack("<Q", len(new_name))
        + new_name
        + struct.pack("<I", target_info[0])
        + b"".join(struct.pack("<Q", d) for d in target_info[1])
        + struct.pack("<I", target_info[2])
        + struct.pack("<Q", target_info[3])  # same offset as token_embd_norm
    )

    new_meta_end = tensor_meta_end + len(new_entry)
    new_data_start = new_meta_end
    if new_data_start % ALIGNMENT != 0:
        new_data_start += ALIGNMENT - (new_data_start % ALIGNMENT)

    shift = new_data_start - old_data_start
    new_padding = new_data_start - new_meta_end

    return {
        "n_tensors": n_tensors,
        "tensor_meta_end": tensor_meta_end,
        "old_data_start": old_data_start,
        "new_data_start": new_data_start,
        "shift": shift,
        "new_entry": new_entry,
        "new_padding": new_padding,
    }


def patch_gguf(path, info):
    file_size = os.path.getsize(path)
    data_size = file_size - info["old_data_start"]
    shift = info["shift"]

    print(f"File size:        {file_size:,} bytes")
    print(f"Data section:     {info['old_data_start']:,} bytes offset, {data_size:,} bytes")
    print(f"Shift amount:     {shift} bytes")
    print(f"New tensor entry: {len(info['new_entry'])} bytes")

    with open(path, "r+b") as f:
        # Step 1: Extend file by shift bytes
        f.seek(0, 2)
        f.write(b"\x00" * shift)
        f.flush()
        print(f"Extended file by {shift} bytes.")

        # Step 2: Shift data section forward (copy backwards to avoid overlap)
        total_moved = 0
        pos = file_size  # read from original end, write to new end
        while pos > info["old_data_start"]:
            chunk_end = pos
            chunk_start = max(info["old_data_start"], pos - CHUNK)
            size = chunk_end - chunk_start

            f.seek(chunk_start)
            data = f.read(size)
            f.seek(chunk_start + shift)
            f.write(data)

            pos = chunk_start
            total_moved += size
            pct = total_moved / data_size * 100
            print(f"\r  Shifting data: {pct:.1f}%", end="", flush=True)

        print(f"\r  Shifting data: 100.0% ({total_moved:,} bytes moved)")

        # Step 3: Write new tensor metadata entry
        f.seek(info["tensor_meta_end"])
        f.write(info["new_entry"])
        f.write(b"\x00" * info["new_padding"])

        # Step 4: Update n_tensors (at byte offset 8, uint64)
        f.seek(8)
        new_n = info["n_tensors"] + 1
        f.write(struct.pack("<Q", new_n))

        f.flush()

    print(f"Done! n_tensors updated to {info['n_tensors']} → {info['n_tensors'] + 1}")
    print(f"output_norm.weight added (same data as token_embd_norm.weight).")


def main():
    path = "/usr/share/ollama/.ollama/models/blobs/sha256-2b1c0ecb28b802cc1c8a8afd42a4746ac9e563e33fe2c87c5948864bda23fe39"

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Check if already patched
    with open(path, "rb") as f:
        f.read(4)  # magic
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
    if n_tensors == 267:
        print("Already patched (n_tensors=267). Nothing to do.")
        return

    print("Scanning GGUF...")
    info = scan_gguf(path)
    print(f"  n_tensors: {info['n_tensors']}")
    print(f"  tensor_meta_end: {info['tensor_meta_end']}")
    print(f"  shift needed: {info['shift']} bytes")

    if info["shift"] == 0:
        print("No shift needed (entry fits in existing padding).")

    patch_gguf(path, info)


if __name__ == "__main__":
    main()
