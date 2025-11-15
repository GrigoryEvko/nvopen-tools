# NVIDIA Fat Binary Format Specification

Reverse-engineered from cuobjdump. Validated against CUDA 11.x-13.x binaries.

## Overview

NVIDIA fat binaries (.nv_fatbin sections) are containers holding multiple GPU code variants (PTX, ELF) for different SM architectures. Entries are stored contiguously with data interleaved. No terminator entry exists - container ends at boundary specified in header.

**Container characteristics:**
- Magic-based detection (0xBA55ED50 or 0x466243B1)
- Contiguous entry+data layout
- ZSTD-compressed payloads
- Multiple containers per file supported
- No alignment requirements
- No terminator entry

## File Structure

### Single Container
```
Offset 0x00: Fat Binary Header (16 bytes)
Offset 0x10: Entry #1 (64/80/112 bytes)
           : Data #1 (ALIGN_8 of compressed size)
           : Entry #2 (64/80/112 bytes)
           : Data #2 (ALIGN_8 of compressed size)
           : ...
           : Entry #N (64/80/112 bytes)
           : Data #N (ALIGN_8 of compressed size)
           : <end - no terminator>
```

### Multiple Containers
A single file may contain thousands of containers. Example: libcublasLt.so.13 contains 2,775 containers with 5,424 ELF entries.

```
[Container #1]
[Container #2]
...
[Container #N]
```

## Fat Binary Header

**Structure (16 bytes):**
```c
struct FatBinaryHeader {
    uint32_t magic;         // 0xBA55ED50 (executable) or 0x466243B1 (relocatable)
    uint16_t version_low;   // 0x0001 (constant)
    uint16_t version_high;  // 0x0010 (first entry offset)
    uint64_t header_size;   // Size of all entries + data (excludes this 16-byte header)
};
```

**Field constraints:**
- magic: Must be 0xBA55ED50 or 0x466243B1
- version_low: Must be 0x0001
- version_high: Must be 0x0010
- header_size: Size from first entry to end (total_size - 16)
- First entry: Always at container_start + 0x10

## Entry Structure

**Base structure (64 bytes):**
```c
struct FatBinaryEntry {
    uint16_t type;                 // 0x01=PTX, 0x02=ELF, 0x10=ELF_ALT, 0x40=unknown
    uint16_t unknown_02;           // Usually 0x0101
    uint32_t size;                 // Entry header size (64/80/112)
    uint32_t aligned_data_size;    // ALIGN_8(compressed_size) with padding
    uint32_t padding;              // Usually 0
    uint32_t data_size;            // Compressed payload size
    uint32_t unknown_14;           // Unknown (0 or 0x40 for extended)
    uint32_t minor_version;        // GPU minor version
    uint32_t sm_arch;              // SM architecture (50-120)
    uint32_t reloc_offset;         // Relocation table offset (usually 0)
    uint32_t reloc_size;           // Relocation table size (usually 0)
    uint64_t flags;                // Processing flags (bit 15 = compression)
    uint8_t  reserved[16];         // Reserved (bytes 8-15 = uncompressed size)
};
```

**Entry sizes:**
- Standard (SM < 100): 64 bytes
- SM 120+ PTX: 80 bytes (64 base + 16 extended)
- SM 100+ ELF: 112 bytes (64 base + 48 extended)

**Entry types:**
- 0x01: PTX code (not counted by cuobjdump --list-elf)
- 0x02: ELF CUBIN (counted by cuobjdump)
- 0x10: ELF CUBIN alternate (counted by cuobjdump)
- 0x40: Unknown (not counted)

## Traversal Algorithm

Entries are traversed contiguously. Container ends when entry_ptr reaches container_end.

**Algorithm:**
```c
uint8_t *entry_ptr = container_start + 0x10;
uint8_t *container_end = container_start + 16 + header_size;

while (entry_ptr + 64 <= container_end) {
    FatBinaryEntry *entry = (FatBinaryEntry *)entry_ptr;

    // Safety check for zeros (not a real terminator)
    if (entry->size == 0) break;

    // Extended header for SM 100+
    uint8_t *extended = entry_ptr + 64;
    size_t extended_size = entry->size - 64;  // 0, 16, or 48 bytes

    // Data location
    uint8_t *data = entry_ptr + entry->size;

    // Advance to next entry
    entry_ptr = entry_ptr + entry->size + entry->aligned_data_size;
}
```

**Key formula:**
```
next_entry = current_entry + entry.size + entry.aligned_data_size
```

**No terminator:**
Container ends exactly after last entry's data. No zero-filled terminator exists.

## Data Layout

Data is embedded immediately after each entry header, with padding to 8-byte alignment:

```
Entry #1 (64 bytes)
  size = 64
  aligned_data_size = 5000 (ALIGN_8(4998))
  data_size = 4998

[Entry #1 header: 64 bytes]
[Entry #1 compressed data: 4998 bytes]
[Padding: 2 bytes of zeros]
[Entry #2 header: 64 bytes]
[Entry #2 compressed data: 3001 bytes]
[Padding: 7 bytes of zeros]
...
```

Data starts at: entry_ptr + entry.size
Next entry starts at: entry_ptr + entry.size + entry.aligned_data_size

Note: aligned_data_size = ALIGN_8(data_size), padding fills the gap.

## Compression

**ELF entries (0x02, 0x10):** Always ZSTD compressed
**PTX entries (0x01):** May be compressed or plain

**ZSTD magic:** 0xFD2FB528 (little-endian: 0x28B52FFD)

**Detection:**
```c
bool is_zstd(uint8_t *data) {
    return *(uint32_t *)data == 0xFD2FB528;
}
```

**Compression flag:** Bit 15 of flags field (0x8000)

**For bit-perfect repacking:**
- Extract compressed data as-is
- Never decompress/recompress
- Original compression settings unknown
- Recompression will not match original

## Packing Algorithm

**Layout calculation:**
```c
uint64_t offset = 16;  // After header

for (int i = 0; i < n; i++) {
    entry[i].size = get_entry_size(entry[i].sm_arch, entry[i].type);
    entry[i].data_size = compressed_size[i];
    entry[i].aligned_data_size = ALIGN_8(compressed_size[i]);
    entry_offset[i] = offset;
    data_offset[i] = offset + entry[i].size;
    offset += entry[i].size + entry[i].aligned_data_size;
}

total_size = offset;
header_size = offset - 16;  // Excludes 16-byte container header
```

**Write sequence:**
```c
write_header(0xBA55ED50, 0x0001, 0x0010, header_size);

for (int i = 0; i < n; i++) {
    write_entry(&entry[i], entry[i].size);     // 64/80/112 bytes
    write_data(&data[i], entry[i].data_size);  // Actual compressed size
    write_padding(entry[i].aligned_data_size - entry[i].data_size);  // Zeros
}

// NO terminator - file ends here
```

## Offset Calculation Example

3 entries with compressed sizes 4998, 3001, 2000 (aligned: 5000, 3008, 2000):

```
Header:      0x0000-0x000F (16 bytes)
Entry 1:     0x0010-0x004F (64 bytes)
Data 1:      0x0050-0x1387 (4998 bytes)
Padding 1:   0x1388-0x1389 (2 bytes)
Entry 2:     0x138A-0x13C9 (64 bytes)
Data 2:      0x13CA-0x1FA1 (3001 bytes)
Padding 2:   0x1FA2-0x1FA8 (7 bytes)
Entry 3:     0x1FA9-0x1FE8 (64 bytes)
Data 3:      0x1FE9-0x27C8 (2000 bytes)
End:         0x27C9

Total: 10185 bytes
header_size: 10169 (excludes 16-byte container header)
```

## Alignment

aligned_data_size uses 8-byte alignment:
```c
#define ALIGN_8(size) (((size) + 7) & ~7UL)
```

Entries can start at any offset (no entry alignment requirement).

## SM Architecture Codes

| Code | Architecture | Example GPUs          |
|------|--------------|----------------------|
| 50   | Maxwell      | GTX 900              |
| 60   | Pascal       | GTX 1080             |
| 70   | Volta        | V100                 |
| 75   | Turing       | RTX 2080             |
| 80   | Ampere       | A100                 |
| 86   | Ampere       | RTX 3090             |
| 89   | Ada          | RTX 4090, L40        |
| 90   | Hopper       | H100, H200           |
| 100  | Blackwell    | B200, B300           |
| 120  | Blackwell    | RTX 5090, RTX 6000   |

## Validation

After packing:
1. Magic bytes correct
2. First entry at offset 0x10
3. Each entry: size ∈ {64, 80, 112}
4. Each entry: aligned_data_size = ALIGN_8(data_size)
5. No terminator (container ends at last entry)
6. header_size = total_size - 16
7. Parseable by cuobjdump

## Edge Cases

**Empty container:** Header only, header_size = 0

**Single entry:** Header + entry + data

**PTX-only:** Container with only type 0x01 entries

**Multiple containers:** Each independent, no cross-references

**Truncated container:** If 16 + header_size > file_size, skip container

**Zero data_size:** Valid for empty entries

## Variable-Length Headers

**Standard (SM < 100):**
```
[64-byte base header]
```

**SM 120+ PTX:**
```
[64-byte base header]
[16-byte extended header]
Total: 80 bytes (entry.size = 80)
```

**SM 100+ ELF:**
```
[64-byte base header]
[48-byte extended header]
Total: 112 bytes (entry.size = 112)
```

Extended header indicated by: entry.size > 64

## Implementation Notes

Correct traversal:
```c
entry_ptr = entry_ptr + entry->size + entry->aligned_data_size;
```

Incorrect traversal:
```c
entry_ptr += 64;  // WRONG - assumes fixed header size
```

Correct layout:
```
[Header][Entry 1 (64/80/112)][Data 1][Padding][Entry 2][Data 2][Padding]...
```

Incorrect assumptions:
- Terminator entry exists (it doesn't)
- Fixed 64-byte headers (can be 80 or 112)
- Data starts at entry + 64 (should be entry + entry.size)
- aligned_data_size is always 64 (it's ALIGN_8 of compressed size)

## References

**Source:** IDA Pro analysis of cuobjdump (CUDA 11.x-13.x)
**Test corpus:** libcublasLt.so.13 (2,775 containers, 5,424 entries)
**Implementation:** fatbin_pack.c, fatbin_unpack.c, fatbin_dump.c
