# NVIDIA Fat Binary Tools

Complete toolkit for NVIDIA .nv_fatbin manipulation with 100% cuobjdump compatibility.

## Tools

### fatbin_dump

Analyze and extract entries from NVIDIA fat binaries.

```bash
# List all entries with architecture distribution
./fatbin_dump nv_fatbin.bin --list-elf

# Extract all entries
./fatbin_dump nv_fatbin.bin --extract-elf all --output-dir /tmp/cubins

# Extract specific architecture
./fatbin_dump nv_fatbin.bin --extract-elf sm_90 --output-dir /tmp/sm90
```

### fatbin_unpack

Extract all entries with detailed manifest for filtering or repacking.

```bash
./fatbin_unpack nv_fatbin.bin /tmp/unpacked
```

Creates:
- `00001_sm75_elf.cubin`, `00002_sm80_elf.cubin`, etc.
- `manifest.txt` with metadata (architecture, type, flags, sizes)

### fatbin_extract_ptx

Extract PTX source code organized by SM architecture.

```bash
./fatbin_extract_ptx nv_fatbin.bin /tmp/ptx_output
```

Output structure:
```
/tmp/ptx_output/
  sm_75/extracted_sm75_00001.ptx
  sm_80/extracted_sm80_00001.ptx
  sm_120/extracted_sm120_00001.ptx
```

Handles ZSTD-compressed and plain PTX automatically.

### fatbin_simple_repack

Quick repacking with ZSTD compression.

```bash
# Default compression (level 3)
./fatbin_simple_repack output.bin *.cubin

# High compression (recommended for distribution)
./fatbin_simple_repack -c 19 output.bin *.cubin

# Maximum compression
./fatbin_simple_repack -c 22 output.bin *.cubin
```

**Compression levels:**
- 1: Fastest
- 3: Default (good balance)
- 9: Better compression
- 19: High compression (87-92% ratio)
- 22: Maximum (marginal gains over 19)

### fatbin_repack

Rebuild from manifest preserving all metadata.

```bash
./fatbin_repack /tmp/unpacked output.bin
```

Reads `manifest.txt` and restores exact entry order, flags, and metadata.

## Build

```bash
make
```

Requires: GCC/Clang, libzstd

## Example Workflow

```bash
# Extract .nv_fatbin section from CUDA library
objcopy --dump-section .nv_fatbin=cublaslt.fatbin libcublasLt.so.13

# Analyze (132 MB fat binary with 2775 containers, 5712 entries)
./fatbin_dump cublaslt.fatbin --list-elf

# Output:
# Found 2775 containers
# Processed 5712 entries
# Found 5424 ELF entries
#
# Architecture Distribution:
#   sm_75  :  167 entries (3.1%)
#   sm_80  :  477 entries (8.8%)
#   sm_90  : 1592 entries (29.4%)
#   sm_100 : 1183 entries (21.8%)
#   sm_120 : 1878 entries (32.9%)

# Unpack all entries
./fatbin_unpack cublaslt.fatbin /tmp/cublaslt_unpacked

# Filter for SM 90 only
cd /tmp/cublaslt_unpacked
ls *_sm90_*.cubin | xargs ../fatbin_simple_repack -c 19 sm90_only.bin

# Result: 1.3 GB → 162 MB (87.9% compression)
```

## cuobjdump Compatibility

**Verified compatible** with NVIDIA cuobjdump from CUDA 11.x-13.x:
- Entry iteration matches exactly
- Supports variable-length headers (SM 100+: 112 bytes, SM 120 PTX: 80 bytes)
- Compression flag handling (bit 15) matches cuobjdump behavior
- Reserved field usage for buffer allocation

**Tested:**
- libcublasLt.so.13 (5712 entries): ✓ All entries extracted
- Compression levels 1-22: ✓ All work
- Round-trip (extract → repack → extract): ✓ Byte-perfect MD5 match

## Format Details

See [FORMAT_SPECIFICATION.md](FORMAT_SPECIFICATION.md) for complete format documentation.

**Key points:**
- Container header: 16 bytes (magic 0xBA55ED50)
- Entry headers: 64/80/112 bytes depending on SM architecture
- Data: ZSTD compressed (magic 0x28B52FFD) or raw
- Entry types: 0x01 (PTX), 0x02 (ELF), 0x04 (LTOIR)
- No terminator entry (last entry ends at container boundary)

## nvFatbin API

Library for programmatic fat binary creation. See [nvFatbin.h](nvFatbin.h) for API reference.

Example:
```c
#include "nvFatbin.h"

nvFatbinHandle handle;
const char *opts[] = {"-compress-level=19"};
nvFatbinCreate(&handle, opts, 1);

nvFatbinAddCubin(handle, cubin_data, cubin_size, "sm_90", NULL);
nvFatbinAddPTX(handle, ptx_code, ptx_size, "sm_120", NULL, NULL);

size_t size;
nvFatbinSize(handle, &size);
void *buffer = malloc(size);
nvFatbinGet(handle, buffer);

nvFatbinDestroy(&handle);
```

Supports all entry types: ELF, PTX, LTOIR, index, relocatable PTX.
