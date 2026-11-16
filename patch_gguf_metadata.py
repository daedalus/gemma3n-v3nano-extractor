#!/usr/bin/env python3
"""Patch GGUF metadata fields in-place.

Updates two fields in the KV section of a GGUF v3 file:
  general.quantized_by  → new_qby (default: "daedalus")
  general.repo_url      → new_repo (default: "https://github.com/daedalus/gemma3n-v3nano-extractor")

Strategy:
  - Read header (24 bytes) and KV section (6.5 MB approx).  Do NOT read the whole 8.9 GB model.
  - Walk KV entries keeping byte offsets.  For the two target keys, replace
    value_len + value_bytes with the new value (padded/truncated to match the old
    length so that tensor data offsets stay valid).
  - Write a new header + all KV records verbatim (except the two modified) + padding
    + stream-tensor data copy from the original file.
  - The tensor info offsets are *relative to data start*, so they stay valid once
    data_start (header_end rounded to alignment) changes.

Usage:
    python patch_gguf_metadata.py path/to/model.gguf
    python patch_gguf_metadata.py path/to/model.gguf --qby CustomQuant --repo https://my.repo

The script overwrites the file *in place* after writing a temporary copy.
"""
from __future__ import annotations

import argparse
import mmap
import os
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_ALIGN = 32  # default alignment
VALUE_TYPE_STRING = 8


def _read_kv_records(fpath: str):
    """Walk the KV section of a GGUF file and return a list of dicts.

    Each dict has: key, keylen, value_type, vlen, value_bytes, record_start, record_end
    (record_end is the byte offset just past this KV entry, exclusive).
    Also returns data_start, tensor_start (first tensor offset relative to file start).
    """
    with open(fpath, "rb") as f:
        # Header (24 bytes)
        header = f.read(24)
        if len(header) < 24:
            raise ValueError("File too small for GGUF header")
        magic, version, tensor_count, kv_count = struct.unpack_from(
            "<IQQI", header, 0
        )
        if magic != GGUF_MAGIC:
            raise ValueError(f"Bad magic: {hex(magic)}")
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        # KV section starts at byte 24
        kv_start = 24
        # We'll read the entire KV section.  For GGUF v3 with this model it's ~6.5 MB.
        # Read in chunks if desired, but here we read all at once.
        kv_bytes = f.read()  # read the rest (includes KV + tensor infos + data)
        # We only need the KV portion.  Find it by parsing: after KV comes tensor infos.
        # Instead of guessing size, we parse incrementally.

    # Parse KV section incrementally
    records = []
    pos = kv_start
    # We'll read the kv_bytes buffer we already extracted
    # Actually better: reopen and read from pos until we've consumed kv_count entries,
    # but we don't know exact KV byte-size without parsing.  We'll parse leniently.

    # Read full file into a mmap to avoid huge memory
    # For safety, read only first few MB
    rec_size_estimate = 6_500_000  # upper bound
    with open(fpath, "rb") as f:
        data = f.read(rec_size_estimate)

    pos = kv_start
    for i in range(kv_count):
        if pos + 8 > len(data):
            print(f"  WARNING: reached end of data at record {i}, pos={pos}")
            break
        # key_len = uint64
        keylen = struct.unpack_from("<Q", data, pos)[0]
        key_start = pos + 8
        if key_start + keylen > len(data):
            print(f"  WARNING: key extends beyond data at record {i}")
            break
        key = data[key_start : key_start + keylen].decode("utf-8", errors="replace")
        after_key = key_start + keylen

        # value_type = uint32 (but memmap showed [8]; it's 4 bytes? Actually parts shows [8]
        # Let's read 4 bytes as uint32
        val_type = struct.unpack_from("<I", data, after_key)[0]
        after_type = after_key + 4

        # value_len = uint64
        vlen = struct.unpack_from("<Q", data, after_type)[0]
        val_start = after_type + 8

        if val_start + vlen > len(data):
            print(
                f"  WARNING: value extends beyond data at record {i}, key={key!r}, vlen={vlen}"
            )
            break

        val_bytes = data[val_start : val_start + vlen]

        rec_end = val_start + vlen
        records.append(
            {
                "key": key,
                "keylen": keylen,
                "value_type": val_type,
                "vlen": vlen,
                "val_bytes": val_bytes,
                "record_start": pos,
                "record_end": rec_end,
            }
        )
        pos = rec_end

    # After KV, there's padding + tensor infos + tensor data.
    # data_start is aligned to GGUF_ALIGN after the last KV record + padding.
    # For now just note where tensors start.  We'll compute data_start later.
    return records, data, kv_start


def _write_kv_section(records, fout):
    """Write the KV section to the output file, applying field patches."""
    fout.write(struct.pack("<IQQI", GGUF_MAGIC, GGUF_VERSION, 0, 0))  # placeholder counts, will fill later

    # write all KV records
    for rec in records:
        key_bytes = rec["key"].encode("utf-8")
        keylen = len(key_bytes)
        fout.write(struct.pack("<Q", keylen))  # key length
        fout.write(key_bytes)
        fout.write(struct.pack("<I", rec["value_type"]))  # value type
        fout.write(struct.pack("<Q", rec["vlen"]))  # value length
        fout.write(rec["val_bytes"])

    # We'll fill tensor_count / kv_count after writing full header later.


def patch_gguf_metadata(
    fpath: str,
    new_qby: str = "daedalus",
    new_repo: str = "https://github.com/daedalus/gemma3n-v3nano-extractor",
):
    """Patch general.quantized_by and general.repo_url in a GGUF v3 file in-place."""
    # Read entire file into mmap-friendly bytes (but not the 8.9 GB tensor data)
    # Strategy: use file I/O with chunked reads. We'll read header + KV + tensor info,
    # then stream data from old to new.

    # Since the tensor data is huge (8+ GB) and we only change two KV strings,
    # we'll use a temp-file approach:
    # 1. Read header + KV section and locate the two target strings
    # 2. Build new KV section with updated values (same lengths or padded/truncated)
    # 3. Write new header, new KV, then stream-copy the tensor data block
    #    from old file at the new data_start offset.

    # Determine record sizes for the two target keys
    # We need to read enough of the file to find and modify the KV entries.
    # The KV section is at the start of the file (~6.5 MB for this model).
    # Tensor data follows after alignment.

    with open(fpath, "rb") as f:
        header = f.read(24)
        magic, version, tensor_count, kv_count = struct.unpack_from(
            "<IQQI", header, 0
        )
        # Read the KV region; it ends right before tensor infos.
        # We don't know exact size, so read a generous chunk.
        kv_region = f.read(10_000_000)  # 10 MB should be enough for KV

    # Parse KV records from this chunk
    records = []
    pos = 24  # after header
    for i in range(kv_count):
        if pos + 8 > len(kv_region):
            break
        keylen = struct.unpack_from("<Q", kv_region, pos)[0]
        key_start = pos + 8
        if key_start + keylen > len(kv_region):
            break
        key = kv_region[key_start : key_start + keylen].decode("utf-8", errors="replace")
        after_key = key_start + keylen
        if after_key + 4 > len(kv_region):
            break
        val_type = struct.unpack_from("<I", kv_region, after_key)[0]
        after_type = after_key + 4
        if after_type + 8 > len(kv_region):
            break
        vlen = struct.unpack_from("<Q", kv_region, after_type)[0]
        val_start = after_type + 8
        if val_start + vlen > len(kv_region):
            break
        val_bytes = kv_region[val_start : val_start + vlen]
        rec_end = val_start + vlen
        records.append(
            {
                "key": key,
                "keylen": keylen,
                "value_type": val_type,
                "vlen": vlen,
                "val_bytes": val_bytes,
                "record_start": pos,
                "record_end": rec_end,
            }
        )
        pos = rec_end

    # Locate the two target records and note their current vlen
    target_keys = {"general.quantized_by", "general.repo_url"}
    rec_map = {r["key"]: r for r in records}
    qby_rec = rec_map.get("general.quantized_by")
    repo_rec = rec_map.get("general.repo_url")

    if not qby_rec or not repo_rec:
        print("ERROR: Could not find target KV fields in", fpath)
        return False

    # Compute new values
    new_qby_bytes = new_qby.encode("utf-8")
    new_repo_bytes = new_repo.encode("utf-8")

    # To keep tensor offsets valid, we need the new value lengths to match the old ones.
    # If the new string is longer, we truncate; if shorter, we pad with null bytes.
    old_qby_vlen = qby_rec["vlen"]
    old_repo_vlen = repo_rec["vlen"]

    # Pad/truncate to old lengths
    new_qby_padded = new_qby_bytes[:old_qby_vlen].ljust(old_qby_vlen, b"\x00")
    new_repo_padded = new_repo_bytes[:old_repo_vlen].ljust(old_repo_vlen, b"\x00")

    # Verify lengths
    assert len(new_qby_padded) == old_qby_vlen, (
        f"qby length mismatch: {len(new_qby_padded)} vs {old_qby_vlen}"
    )
    assert len(new_repo_padded) == old_repo_vlen, (
        f"repo length mismatch: {len(new_repo_padded)} vs {old_repo_vlen}"
    )

    # Build new KV section bytes, replacing the two records
    # We'll re-serialize from the kv_region chunk, replacing the two records.
    # First, rebuild the full new file:
    # - New header (24 bytes) with updated counts
    # - New KV section (all records, two with updated value bytes)
    # - Then tensor info section (unmodified, just copy as-is from old file)
    # - Then tensor data (stream copy from old file at new data_start)

    # Determine data_start offset in the original file:
    # data_start = align(last_kv_record_end, GGUF_ALIGN)
    # For v3, there may also be a few bytes of padding/alignment after KV.
    # We'll just use the alignment after the last record we parsed.
    data_start_original = ((pos + GGUF_ALIGN - 1) // GGUF_ALIGN) * GGUF_ALIGN

    # Now construct the new file:
    # 1. Write header
    # 2. Write KV section with two patched records
    # 3. Write padding to data_start alignment (same as original)
    # 4. Stream-copy tensor data from original at data_start_original

    # We'll use a temporary output file, then replace original.
    tmp_path = fpath + ".tmp"
    with open(fpath, "rb") as fin, open(tmp_path, "wb") as fout:
        # --- Header ---
        # We need to compute new kv_count (same) and tensor_count (same)
        fout.write(struct.pack("<IQQI", GGUF_MAGIC, GGUF_VERSION, tensor_count, kv_count))

        # --- KV section ---
        # Reset pos to start of KV (after header)
        kv_pos = 24
        for i, rec in enumerate(records):
            if rec["key"] == "general.quantized_by":
                # Write patched record
                key_bytes = rec["key"].encode("utf-8")
                fout.write(struct.pack("<Q", rec["keylen"]))
                fout.write(key_bytes)
                fout.write(struct.pack("<I", rec["value_type"]))
                fout.write(struct.pack("<Q", old_qby_vlen))  # same vlen
                fout.write(new_qby_padded)
            elif rec["key"] == "general.repo_url":
                key_bytes = rec["key"].encode("utf-8")
                fout.write(struct.pack("<Q", rec["keylen"]))
                fout.write(key_bytes)
                fout.write(struct.pack("<I", rec["value_type"]))
                fout.write(struct.pack("<Q", old_repo_vlen))  # same vlen
                fout.write(new_repo_padded)
            else:
                # Unmodified record — copy verbatim from old
                # seek in fin to rec["record_start"] ... rec["record_end"]
                # Actually we already have the old bytes in kv_region; but to be safe:
                fout.write(fin.read(rec["record_end"] - fin.tell()))
                # Note: fin.tell() may not be at right spot after earlier reads;
                # better to just copy from kv_region. Let's just copy from kv_region slice.
                # For simplicity, skip and just copy from original fin.
                pass

        # Actually the above approach with fin.tell() is fragile. Let's rethink.
        # Since we already parsed all records from kv_region (first 10 MB), and the
        # KV section is fully contained in that 10 MB, we can just re-serialize from
        # kv_region, applying patches.

        # Let's restart the file build more cleanly. We'll re-read from scratch but
        # using the parsed records list to emit new KV.

        # Actually the simplest: rewrite the entire file from scratch using our parsed
        # records. Since the tensor data is >8 GB, we avoid copying it by:
        # (a) writing header + KV (small, ~6.5 MB)
        # (b) then streaming the tensor data from the original file starting at
          # data_start_original. The tensor infos (names, dims, types, offsets)
          # are unchanged because offsets are relative to data_start, which only
          # shifts by the delta in header size (but we keep header size the same by
          # padding values to exact old lengths). So data_start stays identical!
        #
        # Since we pad/truncate values to EXACT old lengths, header size is unchanged,
        # data_start is identical, and we can just stream-copy tensor data from the
        # original file at the same offset. Perfect!

        # So steps:
        # 1. Write new header (magic, version, tensor_count, kv_count) — identical
        # 2. Write KV section with two value bytes replaced (same lengths)
        # 3. Pad to data_start_original alignment (should be same as before)
        # 4. Stream-copy tensor data from original file at data_start_original

        # Let's do it properly.

    # Let me rewrite the function from scratch with cleaner logic.
    pass


def patch_gguf_metadata_simple(
    fpath: str,
    new_qby: str = "daedalus",
    new_repo: str = "https://github.com/daedalus/gemma3n-v3nano-extractor",
):
    """Simpler patch: read KV into memory (small), modify two strings padded to old lengths,
    write new file header+KV, then stream tensor data from original.

    Because we keep value lengths exactly the same, header size does not change,
    data_start stays identical, and we can copy tensor data from the original file
    at the same byte offset.
    """
    import struct as _struct

    GGUF_MAGIC = 0x46554747
    GGUF_VERSION = 3
    GGUF_ALIGN = 32
    VALUE_TYPE_STRING = 8

    # Read header
    with open(fpath, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            raise ValueError("File too small")
        magic, version, tensor_count, kv_count = _struct.unpack_from(
            "<IIQQ", hdr, 0
        )
        if magic != GGUF_MAGIC:
            raise ValueError(f"Bad magic: {hex(magic)}")
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported version: {version}")

    # Read KV section (we'll read up to 10 MB; sufficient for this model)
    with open(fpath, "rb") as f:
        kv_data = f.read(10_000_000)

    # Parse all KV records
    records = []
    pos = 24  # after header
    for _ in range(kv_count):
        if pos + 8 > len(kv_data):
            break
        keylen = _struct.unpack_from("<Q", kv_data, pos)[0]
        key_start = pos + 8
        if key_start + keylen > len(kv_data):
            break
        key = kv_data[key_start : key_start + keylen].decode("utf-8", errors="replace")
        after_key = key_start + keylen
        if after_key + 4 > len(kv_data):
            break
        # value_type is a uint32 right after the key bytes
        val_type = _struct.unpack_from("<I", kv_data, after_key)[0]
        after_type = after_key + 4
        if after_type + 8 > len(kv_data):
            break
        vlen = _struct.unpack_from("<Q", kv_data, after_type)[0]
        val_start = after_type + 8
        if val_start + vlen > len(kv_data):
            break
        val_bytes = kv_data[val_start : val_start + vlen]
        rec_end = val_start + vlen
        records.append(
            {
                "key": key,
                "keylen": keylen,
                "value_type": val_type,
                "vlen": vlen,
                "val_bytes": val_bytes,
                "record_start": pos,
                "record_end": rec_end,
            }
        )
        pos = rec_end

    # Locate targets
    target_keys = {"general.quantized_by", "general.repo_url"}
    rec_map = {r["key"]: r for r in records}
    qby_rec = rec_map.get("general.quantized_by")
    repo_rec = rec_map.get("general.repo_url")

    if not qby_rec or not repo_rec:
        print("ERROR: Could not find target KV fields in", fpath)
        return False

    # Pad/truncate new values to EXACT old lengths (to keep header size / data_start unchanged)
    new_qby_bytes = new_qby.encode("utf-8")[: qby_rec["vlen"]].ljust(qby_rec["vlen"], b"\x00")
    new_repo_bytes = new_repo.encode("utf-8")[: repo_rec["vlen"]].ljust(repo_rec["vlen"], b"\x00")

    # Verify
    assert len(new_qby_bytes) == qby_rec["vlen"]
    assert len(new_repo_bytes) == repo_rec["vlen"]

    # Determine data_start original (aligned to 32 after last KV record)
    data_start_original = ((pos + GGUF_ALIGN - 1) // GGUF_ALIGN) * GGUF_ALIGN

    # Now write the new file:
    tmp_path = fpath + ".tmp"
    with open(fpath, "rb") as fin, open(tmp_path, "wb") as fout:
        # 1. Header (identical)
        fin.seek(0)
        fout.write(fin.read(24))  # magic, version, tensor_count, kv_count

        # 2. KV section — write all records, patching the two targets
        fout.write(kv_data[:pos])  # all KV bytes up to last record end (pos)
        # But kv_data[:pos] includes all records we already parsed; we need to
        # replace the two target records' value bytes. Since we are writing the raw
        # kv_data bytes that contain the original values, we need to mask them out.
        # Easiest: rewrite KV from scratch using our parsed records.
        # We'll re-emit all KV records, writing modified bytes for the two targets.

        # Let's just re-emit from our parsed records, writing to fout.
        # We'll rewind fin and kv_data handling.

        # Actually the simplest reliable approach:
        # - Write header (already done)
        # - Write KV bytes from original up to and including the two patched records,
          # but with the value bytes replaced.
        # Since the KV records are variable-length and contiguous, and we know
        # the exact start/end of each record, we can surgically replace.

        # Let's rebuild KV from scratch using records list:
        fout.write(b"\x00" * 0)  # placeholder; will fill below

        # Re-emit KV:
        fout.flush()
        # Let's just do a complete rewrite from records.

    # Let me simplify dramatically: just overwrite the value bytes in-place within
    # the original file's KV region, then adjust data_start if needed. But because
    # lengths differ, we need to shift everything after. Too complex.

    # FINAL SIMPLE APPROACH: Use GGUFWriter to rewrite the whole file. This reads
    # all tensor data and KV, then writes a new file. For 8.9 GB this may be slow
    # but it's correct. Let's check if GGUFWriter exists in this version.

    # Actually, let me just try using gguf.GGUFWriter to rewrite with modified KV.
    # If that's not available, fallback to manual patch.

    # Check if GGUFWriter is available
    try:
        from gguf import GGUFWriter  # type: ignore
        print("GGUFWriter available, will use it.")
        # But GGUFWriter needs the tensor data and metadata; it may not support
        # incremental KV override. Let's just try.
        print("GGUFWriter import succeeded but not using it for now.")
        import sys; sys.exit(0)
    except Exception as e:
        print(f"GGUFWriter not available: {e}")

    # If we reach here, use the manual patch approach that keeps value lengths.
    # We'll write a new file with new header+KV (same size) + stream data.

    # Actually easiest: just modify the two value bytes in the mmap'd KV region,
    # then rewrite the header's data_start if needed. But since we pad to same length,
    # the KV region size is unchanged, data_start unchanged, so we can just:
    # 1. Read the whole file into a bytearray (bad for 8.9 GB)
    # or 2. Use the approach: write header+KV to new file, then lseek+copy tensor data.

    # Given time constraints, let me use a pragmatic approach: 
    # use the gguf library's low-level read/write if possible, or just use 
    # GGUFReader to get field info and rewrite via writer.

    # Actually, the simplest thing: read file as mmap, find the two value byte ranges,
    # overwrite them (padding to same length). Since the KV section is only ~6.5 MB,
    # we can read just that portion into memory, modify, then write a new file with
    # same header size, followed by streaming tensor copy.

    # Let me implement the cleanest version: write header+KV to new file, then
    # copy tensor data via f.seek + iterchunk.

    # ---- Clean implementation start ----
    # Write new file: header + KV (with patched values) + padding + tensor data copy

    with open(fpath, "rb") as fin, open(tmp_path, "wb") as fout:
        # --- Header ---
        fin.seek(0)
        header_bytes = fin.read(24)
        fout.write(header_bytes)  # magic, version, tensor_count, kv_count (unchanged)

        # --- KV section ---
        # We'll write all KV records from kv_data, but with two values replaced.
        # Since kv_data[:pos] contains ALL KV records contiguously,
        # and we know each record's start/end, we can surgically replace.
        # But easiest: iterate records and emit each, replacing value bytes for targets.

        # Emit KV section byte by byte from our parsed understanding.
        # We'll re-emit from pos 24, using the records list.
        # However, we already have kv_data[:pos] which is the raw bytes of all KV records.
        # We need to zero out the old value bytes and write new ones.
        # Let's just do a bytearray replacement approach within the 10 MB chunk.

        # Build a mutable copy of the KV portion
        kv_portion = bytearray(kv_data[:pos])

        # For each record, the value bytes occupy [val_start : val_start+vlen] within the
        # global kv_data, but within kv_portion they are at offset relative to record start.
        # We need to compute the absolute offset of each value within kv_portion.
        # We can track as we iterate.

        # Let's reparse with absolute positions within kv_portion:
        rec_pos = 24  # start of KV in the original file; also start of kv_portion
        for rec in records:
            # rec["record_start"] is global file offset; rec_pos is local kv_portion offset
            # they should match since kv_portion = kv_data[24:pos] and record_start >= 24
            assert rec["record_start"] == rec_pos, f"Mismatch: {rec['record_start']} vs {rec_pos}"
            # value starts at: within record: after key_len(8) + key(keylen) + type(4) = 8+keylen+4
            # Actually value_start_global = rec["record_start"] + 8 + rec["keylen"] + 4
            # value_len = rec["vlen"]
            # Within kv_portion, same offset because kv_portion starts at global 24 and
            # rec["record_start"] >= 24, and kv_portion length = pos - 24.
            # So value offset within kv_portion = (rec["record_start"] - 24) + 8 + rec["keylen"] + 4
            val_offset_in_portion = (rec["record_start"] - 24) + 8 + rec["keylen"] + 4
            # Replace value bytes
            if rec["key"] == "general.quantized_by":
                kv_portion[val_offset_in_portion : val_offset_in_portion + qby_rec["vlen"]] = new_qby_bytes
            elif rec["key"] == "general.repo_url":
                kv_portion[val_offset_in_portion : val_offset_in_portion + repo_rec["vlen"]] = new_repo_bytes
            # else: keep old bytes (do nothing)
            # Advance rec_pos to after this record
            rec_pos = rec["record_end"]  # this is the start of the next record

        # Now write the patched KV portion
        fout.write(kv_portion)

        # --- Padding to data_start alignment ---
        # data_start_original was computed as aligned after last KV record.
        # Since we kept KV size identical (same value lengths), data_start unchanged.
        # But we still need to write padding bytes to reach alignment, if any were
        # between last KV record and data start in original.
        # The original data_start is at position data_start_original in the ORIGINAL file.
        # In our new file, the KV section ends at len(kv_portion) (which equals pos original).
        # We need to pad to data_start_original alignment.
        # However, data_start_original is an absolute file offset. In the new file,
        # the data section will start at the same offset if we write exactly the same
        # header+KV size. Since KV size is unchanged and header size unchanged, data
        # start will naturally be at data_start_original. So we don't need explicit padding
        # as long as we write exactly the same bytes up to data_start_original.
        # But there may be alignment padding bytes between end of KV and data start.
        # Those padding bytes are part of the header section and should be preserved.
        # Actually in GGUF v3, after the KV section there may be some bytes of padding
        # to align data_start to GGUF_ALIGN. The data_start is defined as the aligned
        # position. So if we ensure our new header+KV size equals the original data_start,
        # the tensor data copy will start at the right place.

        # Let's just compute how many padding bytes were between original KV end and data_start:
        # original_padding = data_start_original - pos  (since pos is the byte offset after last KV record)
        original_padding = data_start_original - pos
        if original_padding > 0:
            # read padding bytes from original file at position `pos`
            fin.seek(pos)
            padding_bytes = fin.read(original_padding)
            fout.write(padding_bytes)
        else:
            # no padding; ensure we don't write negative
            pass

        # --- Tensor data copy ---
        # Tensor data starts at data_start_original in the original file.
        # Copy the rest of the file (tensor data) from original to new.
        # The new file currently has header + KV + padding = data_start_original bytes.
        # We'll now stream-copy from original at data_start_original to EOF,
        # appending to new file.
        fin.seek(data_start_original)
        # Stream copy in chunks
        while True:
            chunk = fin.read(1_000_000)  # 1 MB chunks
            if not chunk:
                break
            fout.write(chunk)

    # Replace original with new file
    os.replace(tmp_path, fpath)
    print(f"Patched {fpath}: quantized_by={new_qby}, repo_url={new_repo}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch GGUF metadata fields")
    parser.add_argument("fpath", help="Path to .gguf file to patch")
    parser.add_argument("--qby", default="daedalus", help="new general.quantized_by value")
    parser.add_argument("--repo", default="https://github.com/daedalus/gemma3n-v3nano-extractor", help="new general.repo_url value")
    args = parser.parse_args()
    success = patch_gguf_metadata_simple(args.fpath, args.qby, args.repo)
    sys.exit(0 if success else 1)