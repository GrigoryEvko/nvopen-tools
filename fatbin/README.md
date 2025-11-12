# NVIDIA Fat Binary Tools

Parser and extraction tool for NVIDIA fat binary format with complete format documentation.

## Overview

NVIDIA fat binaries are container formats that hold multiple GPU code variants (PTX, ELF cubins) for different SM architectures. This tool provides complete cuobjdump-compatible parsing.

## Usage

```bash
# Build
make

# List all entries
./fatbin_dump nv_fatbin.bin --list-elf

# Extract all entries
./fatbin_dump nv_fatbin.bin --extract-elf all --output-dir /tmp/cubins

# Extract specific architecture
./fatbin_dump nv_fatbin.bin --extract-elf sm_89 --output-dir /tmp/sm89
```

## Example Output

Example from libcublasLt.so.13 (CUDA 13.x):

```
Scanning for fat binary containers...
Found 2775 fat binary containers
Processed 5712 total entries (all types)
Found 5424 ELF entries (type 0x02 + 0x10)

Total entries: 5424

Architecture Distribution:
  sm_75  :  166 entries (3.1%)
  sm_80  :  477 entries (8.8%)
  sm_86  :  100 entries (1.8%)
  sm_89  :  247 entries (4.6%)
  sm_90  : 1592 entries (29.4%)
  sm_100 : 1183 entries (21.8%)
  sm_120 : 1591 entries (29.3%)
```

Note: Entry counts and architecture distribution vary by library and version.

## Fat Binary Format

### File Structure

A file may contain multiple fat binary containers. Example: libcublasLt.so.13 (CUDA 13.x) contains 2,775 containers.

```
+---------------------------+
| Fat Binary Container #1   |
|   - Header (16+ bytes)    |
|   - Entry 1 (64 bytes)    |
|   - Entry 2 (64 bytes)    |
|   - CUBIN data            |
+---------------------------+
| Fat Binary Container #2   |
|   - Header                |
|   - Entries...            |
+---------------------------+
```

### Header Structure (16 bytes)

| Offset | Size | Field       | Description |
|--------|------|-------------|-------------|
| +0x00  | 4    | magic       | 0xBA55ED50 (executable) or 0x466243B1 (relocatable) |
| +0x04  | 2    | version_low | 0x0001 |
| +0x06  | 2    | version_high | 0x0010 (header offset) |
| +0x08  | 8    | header_size | Total container size |

### Entry Structure (64 bytes)

| Offset | Size | Field       | Description |
|--------|------|-------------|-------------|
| +0x00  | 2    | type        | 0x01=PTX, 0x02=ELF, 0x10=ELF alt |
| +0x04  | 4    | size        | Entry structure size |
| +0x08  | 8    | offset      | Offset to CUBIN data |
| +0x10  | 4    | data_size   | Compressed CUBIN size |
| +0x1C  | 4    | sm_arch     | SM architecture (75, 89, 90, etc.) |
| +0x20  | 4    | reloc_offset | Relocation table offset |
| +0x24  | 4    | reloc_size  | Relocation table size |
| +0x28  | 8    | flags       | Processing flags |

### Entry Iteration

Entries use a linked structure:

```c
entry_ptr = fat_binary_start + 0x10;

while (entry_ptr < fat_binary_end) {
    entry = (FatBinEntry*)entry_ptr;

    if (entry->offset == 0 && entry->size == 0)
        break;

    // Process entry...

    // Advance to next entry
    entry_ptr = entry_ptr + entry->offset + entry->size;
}
```

### Type Filtering

Only type 0x02 and 0x10 entries are counted:

- Type 0x01 (PTX): Not counted
- Type 0x02 (ELF): Counted
- Type 0x10 (ELF alt): Counted

### Compression

CUBIN data is ZSTD compressed with magic 0xFD2FB528. Files are extracted as .cubin.zst to preserve exact compression.

## SM Architectures

| Code | Architecture | Hardware |
|------|--------------|----------|
| 75   | Turing       | RTX 20 series |
| 80   | Ampere       | A100 |
| 86   | Ampere       | RTX 30 series |
| 89   | Lovelace     | RTX 40 series |
| 90   | Hopper       | H100 |
| 100  | Blackwell    | Next-gen |

## Implementation

Based on IDA Pro reverse engineering of cuobjdump binary (function at 0x421870). All field offsets and algorithms match cuobjdump exactly.

Verification: Tested against libcublasLt.so.13 (CUDA 13.x) - finds all 5,424 entries that cuobjdump finds.

## Requirements

- GCC or Clang
- libzstd
