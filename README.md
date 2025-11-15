# NVIDIA Open Tools

A collection of tools for working with NVIDIA libraries and binary formats.

## Tools

### fatbin

Fat binary format parser and extraction tool. Reverse engineered from cuobjdump with complete format documentation.

### PTX Prettifier

A comprehensive tool that transforms machine-like PTX assembly into human-readable format with semantic comments, better formatting, and visual code grouping.

#### Features

- **Semantic Comments**: Automatically adds comments explaining what instruction sequences do
- **Better Alignment**: Consistent indentation and operand alignment for readability
- **Visual Grouping**: Separates code sections (parameter loading, thread ID calc, loops, etc.)
- **Pattern Detection**: Identifies common patterns like loops, reductions, synchronization points
- **Instruction Highlighting**: Marks memory operations, control flow, arithmetic ops with descriptive comments
- **Multiple Verbosity Levels**: Choose from minimal to detailed output
- **Preserves Correctness**: Output remains valid, compilable PTX

#### Usage

```bash
# Basic usage - pretty print to stdout
./fatbin/ptx_prettify.py input.ptx

# Save to file
./fatbin/ptx_prettify.py input.ptx -o output_pretty.ptx

# Control verbosity (0=minimal, 1=basic, 2=moderate, 3=detailed)
./fatbin/ptx_prettify.py input.ptx -v 3

# Disable semantic comments
./fatbin/ptx_prettify.py input.ptx --no-comments

# Minimal formatting
./fatbin/ptx_prettify.py input.ptx -v 0 --no-color
```

#### Example Transformation

**Before (machine-like):**
```ptx
ld.param.u64 %rd1, [param_0];
ld.param.u64 %rd2, [param_1];
add.u64 %rd3, %rd1, %rd2;
mov.u32 %r20, %ctaid.x;
setp.eq.s32 %p27, %r11, 0;
@%p27 bra $L__BB0_4;
```

**After (prettified):**
```ptx
═══ Parameter Loading (2 parameters) ═══
    ld.param.u64               %rd1, [param_0]                                 // Load from parameter unsigned 64-bit [param_0]
    ld.param.u64               %rd2, [param_1]                                 // Load from parameter unsigned 64-bit [param_1]
    add.u64                    %rd3, %rd1, %rd2                                // Add operation

═══ Thread/Block ID Calculation ═══
    mov.u32                    %r20, %ctaid.x                                  // Get thread/block index
    setp.eq.s32                %p27, %r11, 0                                   // Set predicate based on condition
    @%p27 bra                        $L__BB0_4                                 // Branch to $L__BB0_4
```

#### Implementation Details

The prettifier consists of several modular components:

1. **PTXParser**: Tokenizes and parses PTX syntax into structured instructions
2. **PatternDetector**: Identifies common code patterns (parameter loading, loops, reductions)
3. **CommentGenerator**: Generates semantic comments based on instruction types
4. **PTXFormatter**: Applies formatting, alignment, and visual grouping

#### Testing

Test on sample PTX files:

```bash
# Test on a single file
python3 fatbin/ptx_prettify.py cublaslt/ptx/sm120/test_fatbin.100.sm_120.ptx -o test_output.ptx

# Batch process multiple files
for f in cublaslt/ptx/sm120/*.ptx; do
    python3 fatbin/ptx_prettify.py "$f" -o "$(basename "$f" .ptx)_pretty.ptx"
done
```
