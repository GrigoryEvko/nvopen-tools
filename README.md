# nvopen-tools

Reverse-engineered tools for NVIDIA binary formats.

## Fat Binary Tools

Complete toolkit for manipulating NVIDIA .nv_fatbin sections found in CUDA libraries.

**Tools:**
- `fatbin_dump` - Analyze and extract entries from fat binaries
- `fatbin_unpack` - Extract all entries with metadata manifest
- `fatbin_extract_ptx` - Extract PTX source code organized by SM architecture
- `fatbin_simple_repack` - Repack cubin files with ZSTD compression (levels 1-22)
- `fatbin_repack` - Rebuild fat binary from manifest preserving metadata

**Features:**
- 100% cuobjdump compatible
- Supports SM 75-121 architectures
- Variable-length headers (64/80/112 bytes)
- ZSTD compression with 87-92% compression ratios
- Handles ELF, PTX, and LTOIR entries
- Tested with 5700+ entry binaries

**Build:**
```bash
cd fatbin
make
```

**Quick Example:**
```bash
# Extract .nv_fatbin from library
objcopy --dump-section .nv_fatbin=output.fatbin libcublasLt.so

# Analyze
./fatbin_dump output.fatbin --list-elf

# Extract all entries
./fatbin_unpack output.fatbin /tmp/extracted

# Repack with better compression
cd /tmp/extracted
ls *.cubin | xargs /path/to/fatbin_simple_repack -c 19 repacked.bin
```

See [fatbin/README.md](fatbin/README.md) for complete documentation.

## Documentation

- [fatbin/FORMAT_SPECIFICATION.md](fatbin/FORMAT_SPECIFICATION.md) - Complete fat binary format documentation
- [fatbin/nvFatbin.h](fatbin/nvFatbin.h) - nvFatbin API reference

## License

MIT
