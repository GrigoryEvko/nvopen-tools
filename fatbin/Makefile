# NVIDIA Fat Binary Tools - Makefile
# ====================================

CC := gcc
CFLAGS := -O2 -Wall -Wextra -std=c11
LDFLAGS := -lzstd

# Targets
# NOTE: fatbin_pack.c is the library implementation, not an executable
TARGETS := fatbin_dump fatbin_unpack fatbin_extract_ptx fatbin_repack fatbin_simple_repack

.PHONY: all clean help install

all: $(TARGETS)

fatbin_dump: fatbin_dump.c
	@echo "Building fatbin_dump..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: fatbin_dump"

fatbin_unpack: fatbin_unpack.c
	@echo "Building fatbin_unpack..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: fatbin_unpack"

fatbin_extract_ptx: fatbin_extract_ptx.c
	@echo "Building fatbin_extract_ptx..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: fatbin_extract_ptx"

fatbin_repack: fatbin_repack.c fatbin_pack.c
	@echo "Building fatbin_repack..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: fatbin_repack"

fatbin_simple_repack: fatbin_simple_repack.c fatbin_pack.c
	@echo "Building fatbin_simple_repack..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓ Built: fatbin_simple_repack"

clean:
	@echo "Cleaning..."
	rm -f $(TARGETS)
	@echo "✓ Clean complete"

test: $(TARGET)
	@echo "Running basic tests..."
	@if [ ! -f /tmp/test_fatbin.bin ]; then \
		echo "Error: No test file at /tmp/test_fatbin.bin"; \
		exit 1; \
	fi
	./$(TARGET) /tmp/test_fatbin.bin --list-elf
	@echo "✓ Tests passed"

install: $(TARGETS)
	@echo "Installing to /usr/local/bin..."
	sudo cp $(TARGETS) /usr/local/bin/
	@echo "✓ Installed $(words $(TARGETS)) tools to /usr/local/bin/"

help:
	@echo "NVIDIA Fat Binary Tools"
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build all tools (default)"
	@echo "  clean       - Remove built binaries"
	@echo "  test        - Run basic tests"
	@echo "  install     - Install to /usr/local/bin"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Tools:"
	@echo "  fatbin_dump          - Parse and analyze fat binaries"
	@echo "  fatbin_unpack        - Extract entries with metadata manifest"
	@echo "  fatbin_extract_ptx   - Extract PTX source from fat binaries"
	@echo "  fatbin_repack        - Rebuild fat binary from manifest"
	@echo ""
	@echo "Library:"
	@echo "  fatbin_pack.c        - nvFatbin API implementation"
	@echo "  nvFatbin.h           - API header file"
	@echo ""
	@echo "Usage Examples:"
	@echo "  ./fatbin_dump input.fatbin --list-elf"
	@echo "  ./fatbin_unpack input.fatbin output_dir/"
	@echo "  ./fatbin_repack output_dir/ repacked.fatbin"
	@echo "  ./fatbin_extract_ptx input.fatbin ptx_output/"
