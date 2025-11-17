# CICC (NVIDIA CUDA Intermediate Code Compiler) - Reverse Engineering

**Status**: L3 Analysis Complete (100% - 27/27 unknowns solved)

## Directory Structure

```
/home/grigory/nvopen-tools/cicc/
├── cicc.i64 (2.6 GB)           # IDA Pro database - CRITICAL
├── cicc_rodata.bin (11 MB)     # Extracted .rodata section
├── decompiled/ (477 MB)         # Hex-Rays decompiled C code (80,281 files)
├── disasm/ (1.1 GB)            # Assembly disassembly (80,562 files)
├── graphs/ (2.2 GB)            # Call graphs (161,124 files)
├── foundation/ (8.4 MB)        # L1 Foundation analysis (150 files)
├── deep_analysis/ (1.3 MB)     # L3 Knowledge extraction (73 files)
├── wiki/ (380 KB)              # Production-ready technical wiki (16 pages)
└── archive/                    # Archived temporary files
    ├── scripts/                # Analysis Python scripts (47 files)
    ├── logs/                   # Execution logs (3 files)
    └── temp_docs/              # Temporary documentation
```

## Quick Start

### View Wiki Documentation
```bash
cd wiki
mkdocs serve  # http://127.0.0.1:8000
```

### Open IDA Database
```bash
ida64 cicc.i64
```

### Access Key Analysis Files
```bash
# Register allocation algorithm
cat deep_analysis/L3/register_allocation/spill_cost_formula.json

# Pattern database (850 patterns)
cat deep_analysis/L3/instruction_selection/pattern_database.json

# All 212 optimization passes
cat deep_analysis/L3/optimization_framework/complete_pass_ordering.json
```

## Analysis Statistics

- **Binary**: 60,108,328 bytes (cicc executable)
- **Functions**: 80,562 total (99.65% decompiled)
- **Analysis Coverage**: 100% (27/27 critical unknowns solved)
- **Knowledge Base**: 1.3 MB (73 files)
- **Wiki Pages**: 16 (ultra-technical, 6,500+ lines)

## Key Discoveries

- **212 Optimization Passes** (indices 10-221)
- **850 IR→PTX Patterns** (3 hash tables: 512, 256, 128)
- **Register Allocation**: K=15, coalesce_factor=0.8, Chaitin-Briggs
- **Tensor Cores**: SM70-SM100 evolution (67→40+→50+ variants)
- **SM100 Features**: FP4 E2M1, 2:4 sparsity, TCGen05
- **Scheduling**: 7 heuristics, DAG construction, critical path

## Documentation

Primary documentation: `/wiki/`
- Ultra-technical reference (zero marketing language)
- All algorithms with C pseudocode
- Exact addresses, constants, and evidence
- 100% coverage of L3 findings

## Archive

Temporary files archived in `archive/`:
- 47 Python analysis scripts
- 3 execution logs  
- 43 temporary documentation files

These are kept for reproducibility but not needed for normal use.

## Git Repository Structure

### Tracked Files (in Git)
- `foundation/` - L1 Foundation analysis (81 MB, 150 files)
- `deep_analysis/` - L3 Knowledge extraction (2.8 MB, 73 files)
- `wiki/` - Production documentation (252 KB, 16 pages)
- `archive/` - Analysis scripts and logs (940 KB)
- `README.md` - This file

### Ignored Files (in .gitignore)
IDA Pro output files are **not tracked** due to size (6.4 GB total):
- `cicc.i64` - IDA database (2.6 GB)
- `decompiled/` - C source (482 MB, 80,281 files)
- `disasm/` - Assembly (1.1 GB, 80,562 files)
- `graphs/` - Call graphs (2.2 GB, 161,124 files)
- `cicc_rodata.bin` - Extracted data (12 MB)

**Note**: To reproduce IDA output, run the analysis scripts in `archive/scripts/` against the original `cicc` binary.
