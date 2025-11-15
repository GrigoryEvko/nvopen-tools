# NVIDIA Fat Binary Format Specification
## Complete Technical Documentation for Bit-Perfect Repacking

*Reverse engineered from cuobjdump, validated against real-world binaries*
*Last updated: 2025-11-12*

---

## Table of Contents
1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Fat Binary Header](#fat-binary-header)
4. [Entry Structure](#entry-structure)
5. [Linked List Traversal](#linked-list-traversal)
6. [Data Embedding](#data-embedding)
7. [Compression](#compression)
8. [Entry Types](#entry-types)
9. [Alignment Requirements](#alignment-requirements)
10. [Edge Cases](#edge-cases)
11. [Packing Algorithm](#packing-algorithm)

---

## Overview

NVIDIA fat binaries are container formats that hold multiple GPU code variants (PTX, ELF cubins) for different SM architectures in a single file. They use a **linked list structure** for entries, not sequential arrays.

**Key Characteristics:**
- Magic-based container detection
- Linked entry structures (NOT sequential!)
- ZSTD-compressed CUBIN data
- Multiple containers per file supported
- Self-describing with size fields

---

## File Structure

### Single Container Layout
```
+---------------------------------------+
| Fat Binary Header (16 bytes)         |  <- Offset 0x0
+---------------------------------------+
| Entry #1 (64 bytes)                   |  <- At header + version_high (0x10)
+---------------------------------------+
| Entry #2 (64 bytes)                   |  <- At entry1_ptr + entry1.offset + entry1.size
+---------------------------------------+
| Entry #3 (64 bytes)                   |
+---------------------------------------+
| ...                                   |
+---------------------------------------+
| Entry #N (64 bytes)                   |
+---------------------------------------+
| Terminator Entry (64 bytes, zeros)    |
+---------------------------------------+
| CUBIN Data for Entry #1 (compressed)  |
+---------------------------------------+
| CUBIN Data for Entry #2 (compressed)  |
+---------------------------------------+
| ...                                   |
+---------------------------------------+
```

### Multiple Containers
A single file may contain **thousands** of fat binary containers:
```
[Container #1: 0x0000 - 0x1234]
[Container #2: 0x1234 - 0x3456]
[Container #3: 0x3456 - 0x5678]
...
[Container #N: ... ]
```

**Example:** `libcublasLt.so.13` contains **2,775 containers** with **5,424 ELF entries** total.

---

## Fat Binary Header

### Structure (16 bytes minimum)
```c
struct FatBinaryHeader {
    uint32_t magic;         // +0x00: 0xBA55ED50 or 0x466243B1
    uint16_t version_low;   // +0x04: Always 0x0001
    uint16_t version_high;  // +0x06: Always 0x0010 (first entry offset)
    uint64_t header_size;   // +0x08: Total container size (bytes)
};
```

### Field Details

| Offset | Size | Field         | Value(s)            | Notes |
|--------|------|---------------|---------------------|-------|
| +0x00  | 4    | magic         | `0xBA55ED50`        | Executable fat binary (most common) |
|        |      |               | `0x466243B1`        | Relocatable fat binary |
| +0x04  | 2    | version_low   | `0x0001`            | Always this value |
| +0x06  | 2    | version_high  | `0x0010`            | Offset to first entry (16 bytes) |
| +0x08  | 8    | header_size   | Variable            | Total size: header + all entries + all data |

### Validation Rules
1. Magic must be `0xBA55ED50` or `0x466243B1`
2. `version_low` must be `0x0001`
3. `version_high` must be `0x0010` (always 16-byte header)
4. `header_size` must be > 16 and <= file size
5. First entry starts at: `container_start + version_high`

---

## Entry Structure

### Complete 64-Byte Layout
```c
struct FatBinaryEntry {
    uint16_t type;              // +0x00: Entry type (0x01, 0x02, 0x10, 0x40)
    uint16_t unknown_02;        // +0x02: Usually 0x0101
    uint32_t size;              // +0x04: Entry structure size (always 64)
    uint64_t offset;            // +0x08: Offset from entry start to data
    uint32_t data_size;         // +0x10: Compressed CUBIN size (bytes)
    uint32_t unknown_14;        // +0x14: Unknown field
    uint32_t minor_version;     // +0x18: GPU minor version
    uint32_t sm_arch;           // +0x1C: SM architecture (50, 75, 89, 90, etc.)
    uint32_t reloc_offset;      // +0x20: Relocation table offset (0 if none)
    uint32_t reloc_size;        // +0x24: Relocation table size (0 if none)
    uint64_t flags;             // +0x28: Processing flags
    uint8_t  reserved[16];      // +0x30: Reserved/padding (zeros)
};
```

### Field Descriptions

| Offset | Size | Field          | Description |
|--------|------|----------------|-------------|
| +0x00  | 2    | type           | Entry type: 0x01=PTX, 0x02=ELF, 0x10=ELF_ALT, 0x40=Unknown |
| +0x02  | 2    | unknown_02     | Usually `0x0101`, purpose unknown |
| +0x04  | 4    | size           | **Always 64** (0x40) - entry structure size |
| +0x08  | 8    | offset         | **Critical:** Offset from entry start to CUBIN data |
| +0x10  | 4    | data_size      | Size of compressed CUBIN data (bytes) |
| +0x14  | 4    | unknown_14     | Unknown field, copy from original |
| +0x18  | 4    | minor_version  | GPU minor version (e.g., 0 for sm_75a, 1 for sm_75b) |
| +0x1C  | 4    | sm_arch        | SM architecture: 50, 75, 80, 86, 89, 90, 100, 120 |
| +0x20  | 4    | reloc_offset   | Relocation table offset (usually 0) |
| +0x24  | 4    | reloc_size     | Relocation table size (usually 0) |
| +0x28  | 8    | flags          | Processing flags, copy from original |
| +0x30  | 16   | reserved       | Reserved space (usually zeros) |

### Entry Type Codes

| Type   | Description        | Counted by cuobjdump? |
|--------|--------------------|-----------------------|
| 0x01   | PTX code           | ❌ No                 |
| 0x02   | ELF CUBIN          | ✅ Yes                |
| 0x10   | ELF CUBIN (alt)    | ✅ Yes                |
| 0x40   | Unknown            | ❌ No                 |

---

## Linked List Traversal

### Critical Algorithm
Entries are **NOT** stored sequentially! They form a linked structure:

```c
// Initialize at first entry
uint8_t *entry_ptr = container_start + version_high;
uint8_t *container_end = container_start + header_size;

// Iterate through linked list
while (entry_ptr + 64 <= container_end) {
    FatBinaryEntry *entry = (FatBinaryEntry *)entry_ptr;

    // TERMINATION CHECK (critical!)
    if (entry->offset == 0 && entry->size == 0) {
        break;  // End of list
    }

    // Validate
    if (entry->size == 0) break;

    // Process entry...
    uint8_t *cubin_data = entry_ptr + entry->offset;

    // ADVANCE TO NEXT ENTRY (key formula!)
    entry_ptr = entry_ptr + entry->offset + entry->size;
}
```

### Key Formula
```
next_entry_ptr = current_entry_ptr + entry->offset + entry->size
```

This means:
1. Jump forward by `entry->offset` bytes (to skip over CUBIN data area)
2. Then add `entry->size` (usually 64) to get to next entry header

### Terminator Entry
The entry list ends with a special terminator:
```c
struct TerminatorEntry {
    // All fields are zero
    uint16_t type;    // 0x0000
    uint16_t unknown; // 0x0000
    uint32_t size;    // 0x0000 (termination marker!)
    uint64_t offset;  // 0x0000 (termination marker!)
    // ... rest zeros
};
```

**Termination condition:** `offset == 0 && size == 0`

---

## Data Embedding

### Data Layout Strategy

The CUBIN data is embedded **between entries** in the linked structure:

```
Entry #1 at offset 0x0010
  ├─ offset = 64 (0x40)
  ├─ size = 64 (0x40)
  └─ data_size = 5000

[64 bytes of Entry #1 header]
[5000 bytes of CUBIN data]      <- Entry #1 data (at entry_ptr + 64)
[64 bytes of Entry #2 header]   <- At entry1_ptr + 64 + 64 = entry1_ptr + 128
[3000 bytes of CUBIN data]
[64 bytes of Entry #3 header]
...
```

### Offset Calculation Formula

For each entry `i`:
```c
// Entry location
entry_offset[i] = entry_offset[i-1] + entry[i-1].offset + entry[i-1].size;

// Data location (relative to entry)
data_offset[i] = 64;  // Data starts immediately after 64-byte header

// Absolute data location in file
data_absolute[i] = entry_offset[i] + data_offset[i];
```

### First Entry Special Case
```c
first_entry_offset = container_start + version_high;  // Usually +0x10
first_data_offset = first_entry_offset + 64;          // Data at +0x50 from start
```

---

## Compression

### ZSTD Compression

**CUBIN entries (type 0x02, 0x10):** ALWAYS ZSTD compressed
**PTX entries (type 0x01):** May or may not be compressed

### ZSTD Frame Format
```
+0x00: 0xFD 0x2F 0xB5 0x28  <- ZSTD magic (little-endian: 0x28B52FFD)
+0x04: Frame Header
  ...
  ZSTD compressed data
  ...
+End: Optional checksum
```

### Compression Detection
```c
bool is_zstd_compressed(uint8_t *data) {
    uint32_t magic = *(uint32_t *)data;
    return (magic == 0xFD2FB528);  // Note: little-endian
}
```

### Compression Settings
**Critical for bit-perfect packing:**

The original compression settings are **unknown**. To achieve bit-perfect output:
1. **Extract compressed data as-is** (keep original .cubin.zst)
2. **Do NOT decompress** during unpacking
3. **Use original compressed data** during repacking
4. **NEVER recompress** - you won't match original compression

If you must recompress (e.g., adding new entries):
- Use ZSTD default compression level (3)
- Single-frame format
- No checksum
- But accept it won't be bit-perfect with original

---

## Entry Types

### Type 0x01: PTX Code
- Human-readable Parallel Thread Execution assembly
- May be compressed or plain text
- NOT counted by `cuobjdump --list-elf`
- Used for forward compatibility (JIT compilation)

### Type 0x02: ELF CUBIN (Primary)
- Binary GPU executable (ELF format)
- Always ZSTD compressed
- Counted by `cuobjdump --list-elf`
- Most common type

### Type 0x10: ELF CUBIN (Alternate)
- Also ELF CUBIN, different variant
- Always ZSTD compressed
- Counted by `cuobjdump --list-elf`
- Purpose: Unknown, possibly debug/optimization variant

### Type 0x40: Unknown
- Purpose unknown
- NOT counted by `cuobjdump --list-elf`
- Rare in practice

---

## Alignment Requirements

### No Strict Alignment
Unlike many binary formats, fat binaries have **NO** strict alignment requirements:

- Entries can start at ANY offset (not aligned to 16, 64, etc.)
- Data can start at ANY offset
- Container can start at ANY offset in file

### Observed Behavior
- First entry: Always at offset 0x10 from container start
- Entry data: Starts at entry + 64 (no padding)
- Entries: Linked by exact arithmetic (no rounding)

### Padding
- Header has NO padding after 16 bytes
- Entries have NO padding after 64 bytes
- Data has NO padding between entries

**Conclusion:** Use exact byte arithmetic, no alignment logic needed.

---

## Edge Cases

### 1. Empty Container
**Q:** Can a container have zero entries?
**A:** Theoretically yes, but extremely rare. Would contain:
```
[16-byte header with header_size = 80]
[64-byte terminator entry with all zeros]
```

### 2. Single Entry Container
Very common, especially in small binaries:
```
[16-byte header]
[64-byte entry #1]
[64-byte terminator]
[CUBIN data]
```

### 3. PTX-Only Container
Container with only PTX entries (type 0x01):
```
[16-byte header]
[64-byte PTX entry]
[64-byte terminator]
[PTX text/compressed data]
```

### 4. data_size vs offset Relationship
```
entry.offset = Always the offset from entry start to data
entry.data_size = Size of the data itself

Example:
  offset = 64 (data is 64 bytes after entry start)
  data_size = 5000 (data is 5000 bytes long)

Next entry is at: entry_ptr + 64 + 64 = entry_ptr + 128
(NOT entry_ptr + 64 + 5000!)
```

### 5. Multiple Containers in Same File
Containers are **independent**:
- Each has its own header
- Each has its own entry list
- No cross-references between containers

### 6. Truncated Containers
**Q:** What if `header_size` exceeds file size?
**A:** Container is invalid, skip it. This can happen with:
- Corrupted binaries
- Accidental magic byte matches in data

### 7. Zero-Size Data
**Q:** Can `data_size` be zero?
**A:** Theoretically yes, but filter these out (invalid entry).

---

## Packing Algorithm

### High-Level Steps

```
1. Read manifest.txt with entry metadata
2. Load compressed CUBIN files (.cubin.zst)
3. Build entry list in memory
4. Calculate offsets using linked-list formula
5. Write header
6. Write entries and data interleaved
7. Write terminator entry
8. Validate output
```

### Detailed Algorithm

```c
// Step 1: Calculate layout
uint64_t current_offset = 16;  // After header

for (int i = 0; i < num_entries; i++) {
    entry[i].offset = 64;  // Data is always 64 bytes after entry
    entry[i].size = 64;    // Entry structure is always 64 bytes

    // Entry at current_offset
    entry_offsets[i] = current_offset;

    // Data at current_offset + 64
    data_offsets[i] = current_offset + 64;

    // Next entry location
    current_offset += 64 + entry[i].data_size;
}

// Terminator entry
terminator_offset = current_offset;
current_offset += 64;

// Total container size
header_size = current_offset;

// Step 2: Write header
write_uint32(0xBA55ED50);     // magic
write_uint16(0x0001);         // version_low
write_uint16(0x0010);         // version_high
write_uint64(header_size);    // header_size

// Step 3: Write entries and data
for (int i = 0; i < num_entries; i++) {
    // Write entry header (64 bytes)
    write_entry(&entry[i]);

    // Write CUBIN data immediately after
    write_cubin_data(&cubin[i]);
}

// Step 4: Write terminator
write_zeros(64);

// Done!
```

### Offset Calculation Example

Given 3 entries with data sizes: 5000, 3000, 2000 bytes

```
Header:        0x0000 - 0x000F (16 bytes)
Entry 1:       0x0010 - 0x004F (64 bytes)
Data 1:        0x0050 - 0x138F (5000 bytes)
Entry 2:       0x1390 - 0x13CF (64 bytes)
Data 2:        0x13D0 - 0x1F8F (3000 bytes)
Entry 3:       0x1F90 - 0x1FCF (64 bytes)
Data 3:        0x1FD0 - 0x27AF (2000 bytes)
Terminator:    0x27B0 - 0x27EF (64 bytes)
Total size:    0x27F0 (10224 bytes)

entry[0].offset = 0x40 (64)
entry[0].size = 0x40 (64)
entry[0].data_size = 0x1388 (5000)

entry[1].offset = 0x40 (64)
entry[1].size = 0x40 (64)
entry[1].data_size = 0xBB8 (3000)

entry[2].offset = 0x40 (64)
entry[2].size = 0x40 (64)
entry[2].data_size = 0x7D0 (2000)
```

### Validation Checks

After packing:
1. ✅ Magic bytes are correct
2. ✅ First entry at offset 0x10
3. ✅ Each entry has offset=64, size=64
4. ✅ Terminator entry at correct location (all zeros)
5. ✅ header_size matches actual file size
6. ✅ Can be parsed by `fatbin_dump --list-elf`
7. ✅ Entry count matches manifest

---

## Common Mistakes to Avoid

### ❌ WRONG: Sequential Entry Layout
```c
// WRONG! Entries are NOT sequential!
entry_ptr += 64;  // Skip to next entry
```

### ✅ CORRECT: Linked List Traversal
```c
// CORRECT! Use linked structure
entry_ptr = entry_ptr + entry->offset + entry->size;
```

### ❌ WRONG: Placing All Entries First
```
[Header]
[Entry 1]
[Entry 2]
[Entry 3]
[Data 1]  <- WRONG! Data is interleaved!
[Data 2]
[Data 3]
```

### ✅ CORRECT: Interleaved Layout
```
[Header]
[Entry 1]
[Data 1]   <- Data immediately after each entry
[Entry 2]
[Data 2]
[Entry 3]
[Data 3]
```

### ❌ WRONG: Recompressing CUBIN Data
```c
// WRONG! This won't match original compression
zstd_compress(cubin_data, level=3);
```

### ✅ CORRECT: Use Original Compressed Data
```c
// CORRECT! Use original .cubin.zst as-is
write(original_compressed_data);
```

### ❌ WRONG: Forgetting Terminator
```c
// WRONG! No terminator means parser runs off end
[Entry N]
[Data N]
<EOF>
```

### ✅ CORRECT: Always Add Terminator
```c
// CORRECT! Terminator signals end of list
[Entry N]
[Data N]
[Terminator: 64 bytes of zeros]
```

---

## Architecture Codes

Common SM architectures:

| Code | Architecture    | GPU Examples                      |
|------|-----------------|-----------------------------------|
| 50   | Maxwell         | GTX 900 series                    |
| 52   | Maxwell         | GTX TITAN X                       |
| 60   | Pascal          | GTX 1080, TITAN X Pascal          |
| 61   | Pascal          | GTX 1050, GTX 1060                |
| 70   | Volta           | Tesla V100                        |
| 75   | Turing          | RTX 2080, TITAN RTX               |
| 80   | Ampere          | A100, A30                         |
| 86   | Ampere          | RTX 3090, RTX 3080                |
| 87   | Ampere          | Orin                              |
| 89   | Ada Lovelace    | RTX 4090, RTX 4080, L40           |
| 90   | Hopper          | H100, H200                        |
| 100  | Blackwell       | B100, B200 (upcoming)             |
| 120  | Future          | TBD                               |

---

## References

### Source Materials
1. IDA Pro reverse engineering of `cuobjdump` binary (function at 0x421870)
2. Analysis of `libcublasLt.so.13` (2,775 containers, 5,424 entries)
3. Testing with various CUDA binaries from CUDA 11.x - 13.x

### Related Tools
- `cuobjdump` - NVIDIA's official tool (closed source)
- `fatbin_dump` - Our open-source parser
- `fatbin_unpack` - Extracts entries with manifest
- `fatbin_extract_ptx` - PTX-specific extractor
- `fatbin_pack` - This specification enables this tool!

### Implementation Notes
For actual packer implementation, see:
- `/home/grigory/nvopen-tools/fatbin/fatbin_pack.c`
- `/home/grigory/nvopen-tools/fatbin/fatbin_unpack.c` (unpacker)
- `/home/grigory/nvopen-tools/fatbin/fatbin_dump.c` (reference parser)

---

**End of Specification**

*This document is sufficient for implementing a bit-perfect fat binary packer from scratch.*
