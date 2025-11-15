# NVIDIA Fat Binary Tools

Parser and extraction tool for NVIDIA fat binary format with complete format documentation.

## Tools

### fatbin_dump

Parse and analyze NVIDIA fat binaries with 100% cuobjdump compatibility.

```bash
# List all entries
./fatbin_dump nv_fatbin.bin --list-elf

# Extract all entries
./fatbin_dump nv_fatbin.bin --extract-elf all --output-dir /tmp/cubins

# Extract specific architecture
./fatbin_dump nv_fatbin.bin --extract-elf sm_89 --output-dir /tmp/sm89
```

### fatbin_unpack

Extract all entries with metadata manifest for analysis or filtering.

```bash
# Unpack to directory with manifest
./fatbin_unpack nv_fatbin.bin /tmp/unpacked
```

### fatbin_extract_ptx

Extract all PTX (Parallel Thread Execution) code from fat binaries, organized by SM architecture.

**Features:**
- Extracts all PTX entries (type 0x01) from fat binaries
- Handles both compressed (ZSTD) and uncompressed PTX
- Automatically decompresses ZSTD-compressed PTX
- Organizes by SM architecture in subdirectories
- Human-readable PTX assembly output

**Usage:**

```bash
# Extract all PTX from a fat binary
./fatbin_extract_ptx nv_fatbin.bin /tmp/ptx_output

# Extract PTX from CUDA library
./fatbin_extract_ptx libcublasLt.so.13 /tmp/cublaslt_ptx
```

**Output Structure:**

```
output_dir/
  sm_75/
    extracted_sm75_00001.ptx
    extracted_sm75_00002.ptx
  sm_80/
    extracted_sm80_00001.ptx
  sm_120/
    extracted_sm120_00001.ptx
```

**PTX Details:**

PTX (Parallel Thread Execution) is NVIDIA's intermediate representation for GPU code:
- Human-readable assembly-like syntax
- Contains .version, .target, .func definitions
- JIT-compiled to SASS (native GPU code) at runtime
- Provides forward compatibility across GPU generations
- Useful for understanding GPU kernel implementations

**Example Output:**

```
Loading input file: libcublasLt.so.13 (124857344 bytes)
Scanning for PTX entries...
Found 2775 fat binary containers
Processed 5712 total entries (all types)
Found 288 PTX entries (type 0x01)

Extracting PTX entries to: /tmp/ptx_output
  Extracted 288/288 PTX entries...

PTX Extraction Statistics:
==========================

Total PTX entries: 288

Architecture Distribution:
  sm_75  : 48 entries (compressed: 48, plain: 0)
  sm_80  : 48 entries (compressed: 48, plain: 0)
  sm_89  : 48 entries (compressed: 48, plain: 0)
  sm_90  : 48 entries (compressed: 48, plain: 0)
  sm_100 : 48 entries (compressed: 48, plain: 0)
  sm_120 : 48 entries (compressed: 48, plain: 0)

Extraction complete!
Successfully extracted: 288/288 PTX entries
Output directory: /tmp/ptx_output
```

### ptx_rename_by_function.py

Rename PTX files based on kernel function names extracted from `.visible .entry` declarations.

**Features:**
- Extracts primary kernel function name from `.visible .entry` declarations
- Renames files to `FUNCTION_NAME.ptx`
- Handles naming conflicts with index suffixes (`_0`, `_1`, `_2`, etc.)
- Skips files with no kernel functions
- Preserves directory structure (in-place renaming)
- Generates JSON manifest with old -> new name mappings
- Supports dry-run mode for safe preview

**Usage:**

```bash
# Preview changes without renaming (dry run)
./ptx_rename_by_function.py /path/to/ptx/dir --dry-run

# Rename files and generate manifest
./ptx_rename_by_function.py /path/to/ptx/dir --manifest renames.json

# Just rename files (no manifest)
./ptx_rename_by_function.py /path/to/ptx/dir

# Help and options
./ptx_rename_by_function.py --help
```

**Example:**

```bash
# Rename PTX files in sm120 directory
./ptx_rename_by_function.py ../cublaslt/ptx/sm120 --dry-run

PTX Function Name Renamer
Directory: /path/to/ptx/sm120

Found 288 PTX files

Renaming 190 files:
--------------------------------------------------------------------------------
test_fatbin.100.sm_120.ptx
  -> sm80_xmma_syrk_nt_u_tilesize32x32x16_stage3_ffma_cp32_kernel.ptx
test_fatbin.101.sm_120.ptx
  -> sm80_xmma_syrk_nt_u_tilesize32x32x16_stage4_ffma_fp32_kernel.ptx
...

SUMMARY
================================================================================
Files to rename: 190
Skipped 98 files:
  test_fatbin.1.sm_120.ptx: No kernel functions found
  ...
```

**PTX Entry Declaration Format:**

PTX files use `.visible .entry` to declare kernel functions:

```ptx
.visible .entry sm80_xmma_syrk_nt_u_tilesize32x32x16_stage3_ffma_cp32_kernel(
    .param .u64 .ptr .align 1 param_0,
    .param .u32 param_1,
    ...
)
```

The tool extracts `sm80_xmma_syrk_nt_u_tilesize32x32x16_stage3_ffma_cp32_kernel` and uses it as the new filename.

## Build

```bash
make
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

Neat detail - you can repack it with much zstd -19 or -22 conversion and get smaller size for free lol.

## Implementation

Based on IDA Pro reverse engineering of cuobjdump binary (function at 0x421870). All field offsets and algorithms match cuobjdump exactly.

Verification: Tested against libcublasLt.so.13 (CUDA 13.x) - finds all 5,424 entries that cuobjdump finds.

## Requirements

- GCC or Clang
- libzstd
